"""Paid API smoke test. Prints client-observed timings, never API credentials."""

import argparse
import asyncio
import json
import time

from dotenv import load_dotenv
from livekit.agents import stt

from livekit import rtc
from livekit.plugins import nari


async def run(args):
    load_dotenv(args.env_file)
    chunk_bytes = 16000 * 2 * args.chunk_ms // 1000
    async with nari.TTS(model=args.tts_model, voice=args.voice) as speaker:
        for index in range(args.runs):
            started = time.perf_counter()
            first = None
            frames = []
            async with speaker.synthesize(args.text) as output:
                async for event in output:
                    if first is None:
                        first = time.perf_counter()
                    frames.append(event.frame)
            if not frames:
                raise RuntimeError("TTS returned no audio")
            synthesis_ms = (time.perf_counter() - started) * 1000
            resampler = rtc.AudioResampler(24000, 16000)
            converted = [f for frame in frames for f in resampler.push(frame)]
            converted.extend(resampler.flush())
            pcm = b"".join(bytes(frame.data) for frame in converted)
            partial_at = None
            committed_at = None
            final_at = None
            transcript = ""
            async with nari.STT(model=args.stt_model) as recognizer:
                async with recognizer.stream() as output:
                    audio_started = time.perf_counter()

                    async def send(pcm=pcm, audio_started=audio_started, output=output):
                        nonlocal committed_at
                        # Wait for capture before sending each chunk; flush the short tail
                        # at its actual end, without padding or waiting for a full chunk.
                        sent = 0
                        for start in range(0, len(pcm), chunk_bytes):
                            chunk = pcm[start : start + chunk_bytes]
                            sent += len(chunk)
                            await asyncio.sleep(
                                max(0, audio_started + sent / 32000 - time.perf_counter())
                            )
                            output.push_frame(
                                rtc.AudioFrame(
                                    data=chunk,
                                    sample_rate=16000,
                                    num_channels=1,
                                    samples_per_channel=len(chunk) // 2,
                                )
                            )
                        committed_at = time.perf_counter()
                        output.end_input()

                    sender = asyncio.create_task(send())
                    try:
                        async with asyncio.timeout(60):
                            async for event in output:
                                if (
                                    event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
                                    and partial_at is None
                                ):
                                    partial_at = time.perf_counter()
                                elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                                    transcript += event.alternatives[0].text
                                    final_at = time.perf_counter()
                        await sender
                    finally:
                        sender.cancel()
                        await asyncio.gather(sender, return_exceptions=True)
            if not transcript.strip() or final_at is None or committed_at is None:
                raise RuntimeError("STT did not return a nonempty final transcript")
            print(
                json.dumps(
                    {
                        "run": index + 1,
                        "framework": "livekit",
                        "tts_model": args.tts_model,
                        "stt_model": args.stt_model,
                        "stt_chunk_ms": args.chunk_ms,  # Caller input, before adapter batching.
                        "stt_wire_chunk_ms": 100,
                        "tts_first_frame_ms": round((first - started) * 1000, 2),
                        "tts_complete_ms": round(synthesis_ms, 2),
                        "audio_seconds": round(len(pcm) / 32000, 3),
                        "stt_first_partial_ms": round((partial_at - audio_started) * 1000, 2)
                        if partial_at
                        else None,
                        "stt_commit_to_final_ms": round((final_at - committed_at) * 1000, 2),
                        "transcript": transcript,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-ms", type=int, choices=(20, 100), default=100)
    parser.add_argument("--tts-model", default="qwen3-tts-fast")
    parser.add_argument("--stt-model", default="qwen3-asr-fast")
    parser.add_argument("--voice", default="diana")
    parser.add_argument("--text", default="Hello, this is a test of Nari speech recognition.")
    asyncio.run(run(parser.parse_args()))
