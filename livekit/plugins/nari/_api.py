"""Small Nari wire client; kept private so adapters can ship independently."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

API_URL = "https://api.narilabs.com/v1"
WS_URL = "wss://api.narilabs.com/v1/realtime?intent=transcription"
STT_CHUNK_BYTES = 3200  # 100 ms of 16 kHz mono PCM16.


class NariError(Exception):
    """Provider error with a support request ID, without request credentials."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: str = "",
        status: int = 0,
        retry_after: float = 0,
    ):
        self.code, self.request_id, self.status = code, request_id, status
        self.retry_after = retry_after
        super().__init__(f"{code}: {message}" + (f" (request {request_id})" if request_id else ""))

    @classmethod
    def from_body(
        cls, body: Any, *, request_id: str = "", status: int = 0, retry_after: float = 0
    ) -> NariError:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
        return cls(
            error.get("code", "API_ERROR"),
            error.get("message", "Nari request failed"),
            request_id=error.get("requestId", request_id),
            status=status,
            retry_after=retry_after,
        )


_TRANSIENT_CODES = frozenset(
    {
        "UPSTREAM_UNAVAILABLE",
        "SERVICE_UNAVAILABLE",
        "AUTH_BACKEND_UNAVAILABLE",
        "RATE_LIMIT_UNAVAILABLE",
        "REQUEST_GATE_UNAVAILABLE",
        "VOICE_CATALOG_UNAVAILABLE",
        "USAGE_QUEUE_UNAVAILABLE",
        "ENTITLEMENT_UNAVAILABLE",
        "SERVER_NOT_READY",
        "SERVER_AT_CAPACITY",
        "SERVER_DRAINING",
        "INTERNAL_ERROR",
        "VAD_OVERLOADED",
        "VAD_UNAVAILABLE",
        "VAD_RUNTIME_ERROR",
    }
)


def error_kind(exc: Exception) -> str:
    """Provider-independent classification; recovery also depends on stream progress."""
    if isinstance(exc, NariError):
        if exc.code == "INVALID_API_KEY" or exc.status == 401:
            return "authentication"
        if exc.code == "INSUFFICIENT_CREDITS" or exc.status == 402:
            return "quota"
        if exc.status == 403:
            return "authorization"
        if exc.status == 429 or exc.code == "RATE_LIMIT_EXCEEDED":
            return "rate_limit"
        if (
            exc.code in {"CONNECTION_CLOSED", "FINAL_TIMEOUT", "SESSION_IDLE_TIMEOUT"}
            or exc.status == 408
        ):
            return "connectivity"
        if 400 <= exc.status < 500:
            return "invalid_request"
        if exc.code in _TRANSIENT_CODES or 500 <= exc.status < 600:
            return "server"
        if exc.code in {
            "INVALID_REQUEST",
            "MODEL_NOT_FOUND",
            "SESSION_CONFIGURATION_LOCKED",
            "MESSAGE_TOO_LARGE",
            "UNSUPPORTED_LANGUAGE",
            "SESSION_SETUP_TIMEOUT",
        }:
            return "invalid_request"
        return "unknown"
    if isinstance(exc, (aiohttp.ClientError, OSError, TimeoutError, ConnectionClosed)):
        return "connectivity"
    if isinstance(exc, ValueError):
        return "invalid_request"
    return "unknown"


def validate_retries(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("max_retries must be a nonnegative integer")


def retry_after_seconds(value: str | None) -> float:
    try:
        return max(0, float(value or "0"))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0


async def retry_wait(exc: Exception, attempt: int, max_retries: int) -> bool:
    """Bounded exponential backoff; never retry sooner than Retry-After."""
    if attempt >= max_retries or error_kind(exc) not in {"server", "connectivity", "rate_limit"}:
        return False
    delay = max(0.25 * 2**attempt + random.uniform(0, 0.1), getattr(exc, "retry_after", 0))
    if delay > 5:  # Leave longer recovery to the application rather than stall the voice turn.
        return False
    await asyncio.sleep(delay)
    return True


def api_key(value: str | None) -> str:
    value = value if value is not None else os.getenv("NARI_API_KEY")
    if not value or not value.strip():
        raise ValueError("Set NARI_API_KEY or pass api_key")
    return value


def split_text(text: str, limit: int = 2048) -> list[str]:
    """Bound Unicode code points, preferring sentence and then word boundaries."""
    text = text.strip()
    result = []
    while len(text) > limit:
        window = text[:limit]
        boundaries = list(re.finditer(r"[.!?。！？](?:\s|$)", window))
        end = boundaries[-1].end() if boundaries else window.rfind(" ")
        if end <= 0:
            end = limit
        result.append(text[:end].strip())
        text = text[end:].strip()
    if text:
        result.append(text)
    return result


async def speech_chunks(
    session: aiohttp.ClientSession,
    *,
    key: str,
    base_url: str,
    text: str,
    model: str,
    voice: str,
    language: str | None,
    seed: int,
    timeout: float,
    connect_timeout: float = 10,
    max_retries: int = 2,
) -> AsyncIterator[tuple[str, bytes]]:
    """Stream whole PCM16 samples. Never retry a partially delivered sentence."""
    validate_retries(max_retries)
    for part in split_text(text):
        for attempt in range(max_retries + 1):
            response_started = False
            try:
                body = dict(
                    model=model,
                    voice=voice,
                    input=part,
                    seed=seed,
                    response_format="pcm",
                    stream=True,
                )
                if language is not None:
                    body["language"] = language
                async with session.post(
                    base_url.rstrip("/") + "/audio/speech",
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(
                        total=None, sock_connect=connect_timeout, sock_read=timeout
                    ),
                ) as response:
                    request_id = response.headers.get("x-request-id", "")
                    if response.status != 200:
                        try:
                            body = await response.json()
                        except (ValueError, aiohttp.ContentTypeError):
                            body = {}
                        raise NariError.from_body(
                            body,
                            request_id=request_id,
                            status=response.status,
                            retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                        )
                    response_started = True
                    if response.content_type != "audio/pcm":
                        raise NariError(
                            "INVALID_AUDIO", "Expected audio/pcm", request_id=request_id
                        )
                    tail = b""
                    received = False
                    async for chunk in response.content.iter_chunked(4800):
                        chunk = tail + chunk
                        size = len(chunk) // 2 * 2
                        tail = chunk[size:]
                        if size:
                            received = True
                            yield request_id, chunk[:size]
                    if tail or not received:
                        raise NariError(
                            "INCOMPLETE_AUDIO",
                            "Empty or truncated PCM16 response",
                            request_id=request_id,
                        )
                break
            except Exception as exc:
                # A 200 response may already have generated/played audio. No resume
                # offset exists, so never retry that request, even before the first yield.
                if response_started or not await retry_wait(exc, attempt, max_retries):
                    raise


async def warm_http(session: aiohttp.ClientSession, *, key: str, base_url: str, model: str) -> None:
    """Establish a reusable HTTP connection without generating billable audio."""
    async with session.get(
        base_url.rstrip("/") + "/voices",
        params={"model": model},
        headers={"Authorization": f"Bearer {key}"},
        allow_redirects=False,
        timeout=aiohttp.ClientTimeout(total=5),
    ) as response:
        if response.status != 200:
            try:
                body = await response.json()
            except (ValueError, aiohttp.ContentTypeError):
                body = {}
            raise NariError.from_body(
                body, status=response.status, request_id=response.headers.get("x-request-id", "")
            )
        await response.read()  # Consume the response so the connection returns to the pool.


class Realtime:
    """One configured session, one receiver, and bounded commit completion waits.

    A reconnect cannot resume a Nari utterance. Errors are surfaced to the caller
    instead of silently dropping or replaying in-flight audio.
    """

    def __init__(
        self,
        *,
        key: str,
        url: str,
        config: dict,
        timeout: float = 10,
        final_timeout: float = 15,
        on_audio_sent: Callable[[bytes], None] | None = None,
        max_retries: int = 2,
    ):
        validate_retries(max_retries)
        self._max_retries = max_retries
        if timeout <= 0 or final_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.key, self.url, self.config = key, url, config
        self._on_audio_sent = on_audio_sent
        self.timeout, self.final_timeout = timeout, final_timeout
        self.ws = None
        self.request_id = ""
        self.failure: Exception | None = None
        self._pending: dict[str, float] = {}
        self._changed = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self._watchdog: asyncio.Task | None = None
        self._sequence = 0
        self._audio_buffer = bytearray()
        self._send_lock = asyncio.Lock()

    async def open(self) -> None:
        for attempt in range(self._max_retries + 1):
            try:
                await self._open_once()
                return
            except Exception as exc:
                if not await retry_wait(exc, attempt, self._max_retries):
                    raise

    async def _open_once(self) -> None:
        self.request_id = ""
        try:
            self.ws = await connect(
                self.url,
                additional_headers={"Authorization": f"Bearer {self.key}"},
                open_timeout=self.timeout,
                close_timeout=2,
                max_size=2**20,
            )
            self.request_id = self.ws.response.headers.get("x-request-id", "")
            await self.ws.send(json.dumps({"type": "session.configure", "session": self.config}))
            event = json.loads(await asyncio.wait_for(self.ws.recv(), self.timeout))
            if event.get("type") == "error":
                raise NariError.from_body(event, request_id=self.request_id)
            if event.get("type") != "session.configured":
                raise NariError("PROTOCOL_ERROR", "Expected session.configured")
            self._watchdog = asyncio.create_task(self._watch_commits())
        except InvalidStatus as exc:
            try:
                body = json.loads(exc.response.body)
            except (ValueError, TypeError):
                body = {}
            raise NariError.from_body(
                body,
                status=exc.response.status_code,
                request_id=exc.response.headers.get("x-request-id", ""),
                retry_after=retry_after_seconds(exc.response.headers.get("Retry-After")),
            ) from None
        except BaseException:
            await self.close()
            raise

    async def append(self, audio: bytes) -> None:
        if not audio:
            return
        if len(audio) % 2:
            raise ValueError("Audio must contain whole PCM16 samples")
        async with self._send_lock:
            if self.failure:
                raise self.failure
            self._audio_buffer.extend(audio)
            while len(self._audio_buffer) >= STT_CHUNK_BYTES:
                chunk = bytes(self._audio_buffer[:STT_CHUNK_BYTES])
                del self._audio_buffer[:STT_CHUNK_BYTES]
                await self._send_audio(chunk)

    async def _send_audio(self, audio: bytes) -> None:
        await self.ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )
        )
        if self._on_audio_sent is not None:
            self._on_audio_sent(audio)

    async def commit(self) -> None:
        # Keep the tail and its commit adjacent even if a VAD callback races
        # with the next input frame. There is no timer or silence padding.
        async with self._send_lock:
            if self.failure:
                raise self.failure
            if self._audio_buffer:
                tail = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                await self._send_audio(tail)
            self._sequence += 1
            event_id = f"commit_{self._sequence}"
            self._pending["commit:" + event_id] = (
                asyncio.get_running_loop().time() + self.final_timeout
            )
            self._update_waiters()
            await self.ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "event_id": event_id})
            )

    def _update_waiters(self) -> None:
        self._changed.set()
        if self._pending:
            self._drained.clear()
        else:
            self._drained.set()

    async def _watch_commits(self) -> None:
        while True:
            self._changed.clear()
            delay = None
            if self._pending:
                delay = max(0, min(self._pending.values()) - asyncio.get_running_loop().time())
            try:
                await asyncio.wait_for(self._changed.wait(), delay)
            except TimeoutError:
                self.failure = NariError(
                    "FINAL_TIMEOUT",
                    "Timed out waiting for a committed transcript",
                    request_id=self.request_id,
                )
                self._drained.set()
                await self.ws.close()
                return

    async def receive(self) -> dict:
        try:
            event = json.loads(await self.ws.recv())
            kind = event.get("type")
            if kind == "error":
                raise NariError.from_body(event, request_id=self.request_id)
            if kind in ("input_audio_buffer.committed", "input_audio_buffer.commit_empty"):
                deadline = self._pending.pop(
                    "commit:" + str(event.get("client_event_id")),
                    asyncio.get_running_loop().time() + self.final_timeout,
                )
                if kind == "input_audio_buffer.committed":
                    self._pending["item:" + event["item_id"]] = deadline
                self._update_waiters()
            elif kind == "transcript.completed":
                self._pending.pop("item:" + event["item_id"], None)
                self._update_waiters()
            return event
        except Exception as exc:
            self.failure = self.failure or exc
            self._drained.set()
            if isinstance(exc, ConnectionClosed) and isinstance(self.failure, ConnectionClosed):
                self.failure = NariError(
                    "CONNECTION_CLOSED",
                    "Realtime session ended; unfinished audio cannot resume",
                    request_id=self.request_id,
                )
            raise self.failure from None

    async def drain(self) -> None:
        await asyncio.wait_for(self._drained.wait(), self.final_timeout)
        if self.failure:
            raise self.failure

    async def close(self) -> None:
        if self._pending and self.failure is None:
            self.failure = NariError(
                "CONNECTION_CLOSED",
                "Session closed before final transcripts",
                request_id=self.request_id,
            )
        self._drained.set()
        # Cancellation discards unsent audio; only an explicit commit flushes it.
        self._audio_buffer.clear()
        if self._watchdog:
            self._watchdog.cancel()
            await asyncio.gather(self._watchdog, return_exceptions=True)
            self._watchdog = None
        if self.ws:
            await self.ws.close()
            self.ws = None
