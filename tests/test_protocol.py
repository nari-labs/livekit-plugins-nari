import asyncio
import base64

import aiohttp
import pytest

from livekit.plugins.nari._api import NariError, Realtime, error_kind, speech_chunks, split_text


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
            await connection.commit()  # Flush a final chunk shorter than 100 ms.
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


async def test_small_frames_form_100ms_messages_and_commit_flushes_tail(server):
    async with server() as peer:
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        await connection.open()
        try:
            await peer.next_event("session.configure")
            frames = [bytes([i, 0]) * 320 for i in range(1, 8)]
            for frame in frames[:4]:
                await connection.append(frame)
            # Less than 100 ms stays local, without sending a premature message.
            await asyncio.sleep(0.01)
            assert peer.events.empty()
            await connection.append(frames[4])
            first = await peer.next_event("input_audio_buffer.append")
            assert base64.b64decode(first["audio"]) == b"".join(frames[:5])
            for frame in frames[5:]:
                await connection.append(frame)
            await connection.commit()
            tail = await peer.next_event("input_audio_buffer.append")
            assert base64.b64decode(tail["audio"]) == b"".join(frames[5:])
            assert (await peer.events.get())["type"] == "input_audio_buffer.commit"
            # A second utterance and an empty commit must not replay the first tail.
            await connection.append(frames[0])
            await connection.commit()
            next_tail = await peer.next_event("input_audio_buffer.append")
            assert base64.b64decode(next_tail["audio"]) == frames[0]
            assert (await peer.events.get())["type"] == "input_audio_buffer.commit"
            await connection.commit()
            assert (await asyncio.wait_for(peer.events.get(), 2))["type"] == (
                "input_audio_buffer.commit"
            )
        finally:
            await connection.close()


async def test_close_discards_unsent_tail(server):
    async with server() as peer:
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        await connection.open()
        await connection.append(b"\1\0" * 320)
        await connection.close()
        assert [event["type"] for _, event in peer.requests] == ["session.configure"]


async def test_tts_retries_rejection_before_success(server):
    async with server() as peer, aiohttp.ClientSession() as http:
        peer.tts_statuses = [503, 200]
        received = [
            data
            async for _, data in speech_chunks(
                http,
                key="test",
                base_url=peer.http_url,
                text="Hello.",
                model="qwen3-tts-fast",
                voice="diana",
                language=None,
                seed=0,
                timeout=1,
            )
        ]
        assert b"".join(received) == peer.pcm
        assert peer.tts_calls == 2


async def test_tts_does_not_retry_a_started_response(server):
    async with server() as peer, aiohttp.ClientSession() as http:
        peer.tts_abort = True
        with pytest.raises(aiohttp.ClientPayloadError):
            async for _ in speech_chunks(
                http,
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
        assert peer.tts_calls == 1


async def test_retry_after_beyond_budget_is_not_retried_early(server):
    async with server() as peer, aiohttp.ClientSession() as http:
        peer.tts_status = 429
        peer.tts_retry_after = "60"
        with pytest.raises(NariError):
            async for _ in speech_chunks(
                http,
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
        assert peer.tts_calls == 1


@pytest.mark.parametrize("status,attempts", [(401, 1), (503, 3)])
async def test_websocket_retries_are_bounded_and_auth_is_permanent(server, status, attempts):
    async with server() as peer:
        peer.ws_statuses = [status] * 4
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        with pytest.raises(NariError) as error:
            await connection.open()
        assert error.value.status == status
        assert error.value.request_id == "ws-rejected"
        assert peer.ws_attempts == attempts
        assert peer.connections == 0
        assert connection.ws is None


async def test_cancelling_backoff_stops_further_attempts(server):
    async with server() as peer:
        peer.ws_statuses = [503] * 4
        connection = Realtime(key="test", url=peer.ws_url, config={"model": "qwen3-asr-fast"})
        task = asyncio.create_task(connection.open())
        async with asyncio.timeout(2):
            while peer.ws_attempts == 0:
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert peer.ws_attempts == 1
        assert connection.ws is None


@pytest.mark.parametrize(
    "status,code,kind",
    [
        (401, "INVALID_API_KEY", "authentication"),
        (402, "INSUFFICIENT_CREDITS", "quota"),
        (403, "API_ERROR", "authorization"),
        (400, "INTERNAL_ERROR", "invalid_request"),
        (429, "API_ERROR", "rate_limit"),
        (503, "API_ERROR", "server"),
        (0, "INSUFFICIENT_CREDITS", "quota"),
        (0, "SERVER_DRAINING", "server"),
        (0, "SESSION_CONFIGURATION_LOCKED", "invalid_request"),
    ],
)
def test_error_classification(status, code, kind):
    assert error_kind(NariError(code, "test", status=status)) == kind
