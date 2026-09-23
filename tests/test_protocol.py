import asyncio
import base64

import aiohttp
import pytest

from livekit.plugins.nari._api import NariError, Realtime, speech_chunks, split_text


def test_text_limit_preserves_words_and_unicode():
    text = "안녕하세요. " * 500 + "End."
    parts = split_text(text)
    assert all(0 < len(part) <= 2048 for part in parts)
    assert " ".join(parts) == text
    assert split_text("😀" * 4097) == ["😀" * 2048, "😀" * 2048, "😀"]


async def test_large_audio_is_split_below_json_limit(server):
    async with server() as peer:
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        await connection.open()
        try:
            audio = b"\0\0" * 100000
            await connection.append(audio)
            count = 0
            while count < len(audio):
                event = await peer.next_event("input_audio_buffer.append")
                assert len(event["audio"]) < 128 * 1024 - 100
                count += len(base64.b64decode(event["audio"]))
            assert count == len(audio)
        finally:
            await connection.close()


async def test_truncated_pcm_response_is_an_error(server):
    async with server() as peer, aiohttp.ClientSession() as session:
        peer.pcm = b"\1\0" * 480 + b"\1"
        with pytest.raises(NariError, match="INCOMPLETE_AUDIO"):
            async for _ in speech_chunks(
                session,
                key="test",
                base_url=peer.http_url,
                text="Hello.",
                model="qwen3-tts-fast",
                voice="diana",
                language=None,
                seed=0,
                timeout=1,
            ):
                pass


async def test_error_event_stops_pending_drain(server):
    async with server() as peer:
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        await connection.open()
        try:
            await peer.socket.send_json(
                {
                    "type": "error",
                    "error": {
                        "code": "INSUFFICIENT_CREDITS",
                        "message": "Add credits",
                        "requestId": "test-id",
                    },
                }
            )
            with pytest.raises(NariError, match="INSUFFICIENT_CREDITS"):
                await connection.receive()
            with pytest.raises(NariError, match="INSUFFICIENT_CREDITS"):
                await asyncio.wait_for(connection.drain(), 0.2)
        finally:
            await connection.close()
