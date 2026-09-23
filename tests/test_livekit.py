import asyncio
import base64

import pytest
from livekit.agents import AgentSession, APIError, UserStateChangedEvent, stt

from livekit import rtc
from livekit.plugins import nari, silero


def audio():
    return rtc.AudioFrame(
        data=b"\x00\x00" * 320, sample_rate=16000, num_channels=1, samples_per_channel=320
    )


async def collect(stream):
    return [event async for event in stream]


async def test_manual_commits_keep_partials_and_wait_for_pending_final(server):
    async with server() as peer:
        peer.final_gate = asyncio.Event()
        async with nari.STT(api_key="test", base_url=peer.ws_url) as provider:
            async with provider.stream() as stream:
                task = asyncio.create_task(collect(stream))
                stream.push_frame(audio())
                stream.flush()
                await peer.next_event("input_audio_buffer.commit")
                stream.end_input()  # Empty commit must not discard the pending final.
                await peer.next_event("input_audio_buffer.commit")
                await asyncio.sleep(0.01)
                assert not task.done()
                peer.final_gate.set()
                events = await asyncio.wait_for(task, 2)
        texts = [e.alternatives[0].text for e in events if e.alternatives]
        assert texts == ["I scream", "Ice cream", "Ice cream is delicious."]
        assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
        assert peer.requests[0][1]["session"]["turn_detection"] is None


async def test_livekit_vad_state_commits_without_ending_user_turn(server):
    async with server() as peer:
        async with nari.STT(api_key="test", base_url=peer.ws_url) as provider:
            session = AgentSession(
                stt=provider, vad=silero.VAD.load(), turn_handling={"turn_detection": "vad"}
            )
            provider.bind(session)
            provider.bind(session)  # Idempotent registration.
            async with provider.stream() as stream:
                stream.push_frame(audio())
                await peer.next_event("session.configure")
                session.emit(
                    "user_state_changed",
                    UserStateChangedEvent(old_state="speaking", new_state="listening"),
                )
                await peer.next_event("input_audio_buffer.commit")
                event = None
                async with asyncio.timeout(2):
                    while event is None or event.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                        event = await anext(stream)
                assert event.alternatives[0].text == "Ice cream is delicious."


async def test_final_timeout_is_bounded_and_does_not_replay_audio(server):
    async with server() as peer:
        peer.final_gate = asyncio.Event()
        async with nari.STT(api_key="test", base_url=peer.ws_url, final_timeout=0.05) as provider:
            async with provider.stream() as stream:
                stream.push_frame(audio())
                stream.end_input()
                with pytest.raises(APIError):
                    await asyncio.wait_for(collect(stream), 2)
        assert peer.connections == 1


async def test_duration_boundary_does_not_emit_end_of_speech(server):
    async with server() as peer:
        async with nari.STT(api_key="test", base_url=peer.ws_url) as provider:
            async with provider.stream() as stream:
                stream._handle_event(
                    {
                        "type": "transcript.completed",
                        "item_id": "long",
                        "transcript": "still speaking",
                        "commit_reason": "max_duration",
                    },
                    "test",
                )
                assert (await anext(stream)).type == stt.SpeechEventType.START_OF_SPEECH
                assert (await anext(stream)).type == stt.SpeechEventType.FINAL_TRANSCRIPT
                stream.end_input()
                rest = await asyncio.wait_for(collect(stream), 2)
                assert not any(e.type == stt.SpeechEventType.END_OF_SPEECH for e in rest)


@pytest.mark.parametrize("streaming", [False, True])
async def test_first_audio_arrives_before_http_or_text_completion(server, streaming):
    async with server() as peer:
        peer.audio_gate = asyncio.Event()
        async with nari.TTS(api_key="test", base_url=peer.http_url) as provider:
            output = provider.stream() if streaming else provider.synthesize("Hello.")
            async with output:
                if streaming:
                    output.push_text("Hello. Next")  # Not end_input; LLM is still generating.
                first = await asyncio.wait_for(anext(output), 2)
                assert len(first.frame.data) > 0
                assert first.frame.sample_rate == 24000
                assert not peer.tts_complete
                peer.audio_gate.set()
                if streaming:
                    output.end_input()
                rest = [event async for event in output]
                assert rest
        assert all(body["stream"] is True for _, body in peer.requests)
        assert all(body["response_format"] == "pcm" for _, body in peer.requests)


async def test_tts_preserves_pcm_sample_boundaries(server):
    async with server() as peer:
        async with nari.TTS(api_key="test", base_url=peer.http_url) as provider:
            async with provider.synthesize("Hello.") as stream:
                pcm = b"".join([bytes(event.frame.data) async for event in stream])
        # LiveKit may add a final 10 ms silence marker; actual samples must be intact.
        assert pcm.startswith(peer.pcm)
        assert set(pcm[len(peer.pcm) :]) <= {0}


async def test_tts_cancel_closes_request_without_waiting_for_body(server):
    async with server() as peer:
        peer.audio_gate = asyncio.Event()
        async with nari.TTS(api_key="test", base_url=peer.http_url) as provider:
            stream = provider.synthesize("Hello.")
            await asyncio.wait_for(anext(stream), 2)
            await asyncio.wait_for(stream.aclose(), 1)
            assert stream.done
            assert not peer.tts_complete


async def test_auth_or_credit_error_is_not_retried(server):
    async with server() as peer:
        peer.tts_status = 402
        async with nari.TTS(api_key="test", base_url=peer.http_url) as provider:
            async with provider.synthesize("Hello.") as stream:
                with pytest.raises(APIError, match="INSUFFICIENT_CREDITS"):
                    await collect(stream)
        assert peer.tts_calls == 1


async def test_stream_batches_input_but_flush_and_end_preserve_utterances(server):
    async with server() as peer:
        async with nari.STT(api_key="test", base_url=peer.ws_url) as provider:
            async with provider.stream() as stream:
                frames = [bytes([i, 0]) * 320 for i in range(10)]
                for index, data in enumerate(frames):
                    stream.push_frame(
                        rtc.AudioFrame(
                            data=data, sample_rate=16000, num_channels=1, samples_per_channel=320
                        )
                    )
                    if index == 6:
                        stream.flush()
                stream.end_input()
                events = await asyncio.wait_for(collect(stream), 2)
        wire = [e for _, e in peer.requests if e["type"] != "session.configure"]
        assert [e["type"] for e in wire] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        audio = [base64.b64decode(e["audio"]) for e in wire if "audio" in e]
        assert [len(chunk) for chunk in audio] == [3200, 1280, 1920]
        assert b"".join(audio) == b"".join(frames)
        assert sum(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in events) == 2
