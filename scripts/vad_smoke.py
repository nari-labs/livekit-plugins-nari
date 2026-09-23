"""Paid headless smoke test: native TTS stream, real Silero VAD, Nari STT."""

import argparse
import asyncio
import json

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession
from livekit.agents.voice.io import AudioInput

from livekit import rtc
from livekit.plugins import nari, silero


async def main(args):
    load_dotenv(args.env_file)
    async with nari.TTS() as speaker:
        async with speaker.stream() as output:
            output.push_text("Hello. This is a test of Nari speech recognition.")
            output.end_input()
            frames = [event.frame async for event in output]
    resampler = rtc.AudioResampler(24000, 16000)
    converted = [f for frame in frames for f in resampler.push(frame)]
    converted.extend(resampler.flush())
    pcm = b"\0" * 16000 + b"".join(bytes(frame.data) for frame in converted) + b"\0" * 64000
    ready = asyncio.Event()
    finished = asyncio.Event()
    events = []

    async def audio():
        await ready.wait()
        for start in range(0, len(pcm), 640):
            chunk = pcm[start : start + 640]
            yield rtc.AudioFrame(
                data=chunk, sample_rate=16000, num_channels=1, samples_per_channel=len(chunk) // 2
            )
            await asyncio.sleep(0.02)
        await finished.wait()

    recognizer = nari.STT()
    session = AgentSession(
        stt=recognizer,
        vad=silero.VAD.load(min_silence_duration=0.2),
        turn_handling={"turn_detection": "vad", "interruption": {"enabled": False}},
    )
    recognizer.bind(session)

    class Input(AudioInput):
        def __init__(self):
            super().__init__(label="paced-smoke")
            self.stream = audio()

        async def __anext__(self):
            return await anext(self.stream)

    session.input.audio = Input()
    final = asyncio.Event()

    @session.on("user_state_changed")
    def state(event):
        events.append({"state": event.new_state})

    @session.on("user_input_transcribed")
    def transcribed(event):
        events.append({"text": event.transcript, "final": event.is_final})
        if event.is_final and event.transcript.strip():
            final.set()

    try:
        await session.start(
            agent=Agent(instructions="Transcribe."), record=False, session_host=False
        )
        ready.set()
        await asyncio.wait_for(final.wait(), 20)
        assert any(e.get("state") == "speaking" for e in events)
        assert any(e.get("state") == "listening" for e in events)
        print(json.dumps(events))
    finally:
        finished.set()
        await session.aclose()
        await recognizer.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    asyncio.run(main(parser.parse_args()))
