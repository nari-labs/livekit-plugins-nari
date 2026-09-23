# Validation — 2026-09-23

Environment: Linux, Python 3.12.13, LiveKit Agents 1.8.2. API credentials were
loaded from an external `.env`; no credentials are included in this repository.

## Real Nari API

Requested TTS model: `qwen3-tts-fast`, voice: `diana`.
Requested STT session model: `qwen3-asr-fast`, manual commit, 16 kHz PCM16 mono.
The smoke test used actual plugin output and streamed generated audio at 20 ms
microphone pacing, concurrently consuming transcripts.

Phrase: "Hello, this is a test of Nari speech recognition."

| Run | First TTS frame (ms) | Complete TTS (ms) | Commit to final STT (ms) |
| --- | ---: | ---: | ---: |
| 1, cold HTTP connection | 248.44 | 452.04 | 116.09 |
| 2, reused HTTP connection | 96.95 | 297.44 | 115.41 |
| 3, reused HTTP connection | 97.10 | 296.81 | 118.66 |

All three recognized: "Hello. This is a test of Nari speech recognition."
Generated audio was about 3 seconds. A further 14.3-second utterance was also
recognized correctly: first TTS frame 240.18 ms on a new connection, complete
TTS 1204.82 ms, STT commit to final 197.78 ms.

**No partial transcript events were observed in these live runs**, including
the longer utterance. The public protocol permits finals without partials.
Partial-event translation is covered by the local WebSocket tests; its delivery
from the deployed API remains unverified. The adapter does not wait for partials.

These small samples are smoke tests, not p50/p95/p99 benchmarks or server-only
latency claims. Timings include network and client processing. STT measurements
exclude the utterance duration and VAD/semantic endpointing; TTS measurements
exclude device playback and LLM sentence generation.

## Headless LiveKit VAD acceptance

`scripts/vad_smoke.py` also passed against the live API. It synthesized speech
through the native text-stream TTS path, fed paced PCM plus silence into a real
AgentSession with Silero VAD, and observed `speaking -> listening -> final
transcript` without manually flushing STT or injecting speech-state events.
The recognized phrase matched the input. No LLM or room transport was involved.

## Automated verification

- Local real HTTP/WebSocket peer: configuration before audio, revisable partials,
  delayed finals, empty commits, bounded missing-final waits, service errors.
- Public LiveKit session-event binding and 36-second duration-boundary mapping.
- First audio available before both HTTP completion and LLM text completion.
- Odd HTTP chunk boundaries preserve PCM16 samples; truncated audio fails.
- Cancellation closes the in-flight synthesis operation; credit errors do not retry.
- Package build and example CLI import/startup argument parsing.

## Remaining acceptance work

- Full microphone/speaker conversation and room transport playback.
- Semantic turn detector behavior with real multi-turn speech.
- End-to-end interruption audibility in a deployed agent.
- Partial transcripts from the deployed API, if that deployment enables them.
- Broader language/voice coverage and statistically meaningful load benchmarks.
