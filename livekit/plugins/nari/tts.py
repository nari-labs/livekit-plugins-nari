"""HTTP audio streaming with LiveKit's sentence aggregation and cancellation."""

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from contextlib import aclosing

import aiohttp
from livekit.agents import APIConnectOptions, tokenize, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from ._api import API_URL, speech_chunks, warm_http
from ._api import api_key as resolve_api_key
from ._errors import api_error


class TTS(tts.TTS):
    """Synthesize complete text, streaming 24 kHz PCM as it arrives.

    Text streaming is adapted to sentence requests locally. A native stream
    avoids LiveKit's generic adapter adding a second 200 ms audio frame buffer.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen3-tts-fast",
        voice: str = "diana",
        language: str | None = None,
        seed: int = 0,
        base_url: str = API_URL,
        read_timeout: float = 30,
        sentence_tokenizer: tokenize.SentenceTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        self._key = resolve_api_key(api_key)
        self._model, self._voice, self._language, self._seed = model, voice, language, seed
        self._url, self._timeout = base_url, read_timeout
        self._session = http_session
        self._owns_session = http_session is None
        self._tokenizer = sentence_tokenizer or tokenize.blingfire.SentenceTokenizer(
            min_sentence_len=1, stream_context_len=1, max_token_len=2048, retain_format=True
        )
        self._streams: weakref.WeakSet = weakref.WeakSet()
        self._warmup_task: asyncio.Task[None] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Nari Labs"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        if not text.strip():
            raise ValueError("Text must not be empty")
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def _http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def _audio(self, text: str, connect_timeout: float, max_retries: int):
        try:
            async with aclosing(
                speech_chunks(
                    self._http_session(),
                    key=self._key,
                    base_url=self._url,
                    text=text,
                    model=self._model,
                    voice=self._voice,
                    language=self._language,
                    seed=self._seed,
                    timeout=self._timeout,
                    connect_timeout=connect_timeout,
                    max_retries=max_retries,
                )
            ) as chunks:
                async for item in chunks:
                    yield item
        except Exception as exc:
            raise api_error(exc) from exc

    def prewarm(self) -> None:
        """Prepare the HTTP connection in the background; no speech is synthesized."""
        if self._warmup_task is None or self._warmup_task.done():
            self._warmup_task = asyncio.create_task(
                warm_http(
                    self._http_session(), key=self._key, base_url=self._url, model=self._model
                )
            )
            self._warmup_task.add_done_callback(self._warmup_done)

    @staticmethod
    def _warmup_done(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logging.getLogger(__name__).warning("Nari connection warmup failed: %s", error)

    async def warmup(self) -> None:
        """Await connection preparation, surfacing errors to an explicit caller."""
        self.prewarm()
        assert self._warmup_task is not None
        await asyncio.shield(self._warmup_task)

    async def aclose(self) -> None:
        if self._warmup_task is not None:
            self._warmup_task.cancel()
            await asyncio.gather(self._warmup_task, return_exceptions=True)
            self._warmup_task = None
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None


class ChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._tts
        initialized = False
        async with aclosing(
            provider._audio(
                self.input_text, self._conn_options.timeout, self._conn_options.max_retry
            )
        ) as chunks:
            async for request_id, chunk in chunks:
                if not initialized:
                    output_emitter.initialize(
                        request_id=request_id,
                        sample_rate=24000,
                        num_channels=1,
                        mime_type="audio/pcm",
                        frame_size_ms=20,
                    )
                    initialized = True
                output_emitter.push(chunk)
                # Flush partial frames too: don't wait for 200 ms of generated audio.
                output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        provider = self._tts
        sentences = provider._tokenizer.stream()
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )

        async def forward_text():
            async for value in self._input_ch:
                if isinstance(value, self._FlushSentinel):
                    sentences.flush()
                else:
                    sentences.push_text(value)
            sentences.end_input()

        async def synthesize():
            current_segment = None
            async for sentence in sentences:
                if not sentence.token.strip():
                    continue
                self._mark_started()
                async with aclosing(
                    provider._audio(
                        sentence.token, self._conn_options.timeout, self._conn_options.max_retry
                    )
                ) as chunks:
                    async for request_id, audio in chunks:
                        if current_segment != sentence.segment_id:
                            if current_segment is not None:
                                output_emitter.end_segment()
                            output_emitter.start_segment(segment_id=request_id or uuid.uuid4().hex)
                            current_segment = sentence.segment_id
                        output_emitter.push(audio)
                        output_emitter.flush()
            if current_segment is not None:
                output_emitter.end_segment()

        tasks = [asyncio.create_task(forward_text()), asyncio.create_task(synthesize())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await sentences.aclose()
