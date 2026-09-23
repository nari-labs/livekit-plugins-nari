"""Small Nari wire client; kept private so adapters can ship independently."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

API_URL = "https://api.narilabs.com/v1"
WS_URL = "wss://api.narilabs.com/v1/realtime?intent=transcription"
STT_CHUNK_BYTES = 3200  # 100 ms of 16 kHz mono PCM16.


class NariError(Exception):
    """Provider error with a support request ID, without request credentials."""

    def __init__(self, code: str, message: str, *, request_id: str = "", status: int = 0):
        self.code, self.request_id, self.status = code, request_id, status
        super().__init__(f"{code}: {message}" + (f" (request {request_id})" if request_id else ""))

    @classmethod
    def from_body(cls, body: Any, *, request_id: str = "", status: int = 0) -> NariError:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
        return cls(
            error.get("code", "API_ERROR"),
            error.get("message", "Nari request failed"),
            request_id=error.get("requestId", request_id),
            status=status,
        )


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
) -> AsyncIterator[tuple[str, bytes]]:
    """Stream whole PCM16 samples. Never retry a partially delivered sentence."""
    for part in split_text(text):
        body = dict(
            model=model, voice=voice, input=part, seed=seed, response_format="pcm", stream=True
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
                raise NariError.from_body(body, request_id=request_id, status=response.status)
            if response.content_type != "audio/pcm":
                raise NariError("INVALID_AUDIO", "Expected audio/pcm", request_id=request_id)
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
                    "INCOMPLETE_AUDIO", "Empty or truncated PCM16 response", request_id=request_id
                )


class Realtime:
    """One configured session, one receiver, and bounded commit completion waits.

    A reconnect cannot resume a Nari utterance. Errors are surfaced to the caller
    instead of silently dropping or replaying in-flight audio.
    """

    def __init__(
        self, *, key: str, url: str, config: dict, timeout: float = 10, final_timeout: float = 15
    ):
        if timeout <= 0 or final_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.key, self.url, self.config = key, url, config
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
        # Cancellation discards unsent audio; only an explicit commit flushes it.
        self._audio_buffer.clear()
        if self._watchdog:
            self._watchdog.cancel()
            await asyncio.gather(self._watchdog, return_exceptions=True)
            self._watchdog = None
        if self.ws:
            await self.ws.close()
            self.ws = None
