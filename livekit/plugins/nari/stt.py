"""Realtime speech recognition backed by Nari's WebSocket API."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any

from livekit.agents import AgentSession, APIConnectOptions, APIError, UserStateChangedEvent, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from livekit import rtc

from ._api import WS_URL, Realtime
from ._api import api_key as resolve_api_key


class STT(stt.STT):
    """Streaming STT with LiveKit-owned VAD and turn handling.

    Call ``bind(session)`` before starting an AgentSession with a local VAD.
    Its speaking-to-listening events commit transcription, while LiveKit keeps
    ownership of semantic turn detection and interruptions. Standalone clients
    call ``stream.flush()`` explicitly. Server VAD remains an opt-in alternative.
    STT transmission batches into 100 ms chunks and flushes the tail on commit.
    Create one STT instance per AgentSession.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen3-asr-fast",
        language: str | None = None,
        prompt: str = "",
        turn_detection: dict[str, Any] | None = None,
        base_url: str = WS_URL,
        final_timeout: float = 15,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        self._key = resolve_api_key(api_key)
        self._model, self._language = model, language
        self._config = dict(
            model=model,
            language=language,
            prompt=prompt,
            word_timestamps=False,
            turn_detection=turn_detection,
        )
        self._url, self._final_timeout = base_url, final_timeout
        self._streams: weakref.WeakSet[RecognizeStream] = weakref.WeakSet()
        self._bound_session: AgentSession | None = None

    def bind(self, session: AgentSession) -> None:
        """Connect the session's public speech-state events to Nari commits.

        Use LiveKit VAD or its turn detector, not STT endpointing mode. This
        hook finalizes an ASR segment; it never calls commit_user_turn().
        """
        if self._config["turn_detection"] is not None:
            raise ValueError("bind() requires manual Nari commits (turn_detection=None)")
        if session.stt is not self or session.vad is None:
            raise ValueError("Pass this STT and a VAD to AgentSession before calling bind()")
        if session.turn_detection == "stt":
            raise ValueError("Use LiveKit VAD or a turn detector with bind(), not STT endpointing")
        if self._bound_session is session:
            return
        if self._bound_session is not None:
            raise ValueError("Create one Nari STT instance per AgentSession")
        self._bound_session = session
        session.on("user_state_changed", self._on_user_state_changed)

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        if event.old_state == "speaking" and event.new_state == "listening":
            for stream in list(self._streams):
                try:
                    stream.flush()
                except RuntimeError:
                    # A stream may have ended during an agent handoff.
                    continue

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Nari Labs"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Use STT.stream(); Nari exposes realtime transcription")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        stream = RecognizeStream(self, language=language, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        if self._bound_session is not None:
            self._bound_session.off("user_state_changed", self._on_user_state_changed)
            self._bound_session = None
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))


class RecognizeStream(stt.RecognizeStream):
    def __init__(
        self, provider: STT, *, language: NotGivenOr[str], conn_options: APIConnectOptions
    ):
        self._provider = provider
        self._config = dict(provider._config)
        if is_given(language):
            self._config["language"] = language
        self._speaking = False
        super().__init__(stt=provider, conn_options=conn_options, sample_rate=16000)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.num_channels != 1:
            raise ValueError("Nari STT requires mono audio")
        super().push_frame(frame)

    async def _run(self) -> None:
        connection = Realtime(
            key=self._provider._key,
            url=self._provider._url,
            config=self._config,
            timeout=self._conn_options.timeout,
            final_timeout=self._provider._final_timeout,
        )
        tasks = []
        try:
            await connection.open()

            async def send() -> None:
                async for frame in self._input_ch:
                    if isinstance(frame, self._FlushSentinel):
                        await connection.commit()
                    else:
                        await connection.append(bytes(frame.data))
                await connection.drain()

            async def receive() -> None:
                while True:
                    self._handle_event(await connection.receive(), connection.request_id)

            sender = asyncio.create_task(send())
            receiver = asyncio.create_task(receive())
            tasks = [sender, receiver]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Replaying or skipping consumed audio on a fresh session would be
            # lossy. Let the application decide how to recover instead.
            raise APIError(str(exc), retryable=False) from exc
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await connection.close()

    def _handle_event(self, event: dict, request_id: str) -> None:
        kind = event["type"]
        text = event.get("transcript", "")
        if kind == "input_audio_buffer.speech_started" or (
            text and kind in ("transcript.partial", "transcript.completed")
        ):
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, request_id=request_id)
                )
        if kind in ("transcript.partial", "transcript.completed"):
            final = kind == "transcript.completed"
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT
                    if final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(
                            language=event.get("language") or self._config.get("language") or "und",
                            text=text,
                            metadata={
                                "item_id": event["item_id"],
                                "commit_reason": event.get("commit_reason"),
                            },
                        )
                    ],
                )
            )
            # A duration boundary finalizes a segment, not the user's turn.
            if final and event.get("commit_reason") != "max_duration":
                self._speaking = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, request_id=request_id)
                )
