# Validation — 2026-09-23

Environment: Linux, Python 3.12.13, LiveKit Agents 1.8.2. API credentials were
loaded from an external `.env`; no credentials are included in this repository.

## Initial real Nari API test (20 ms chunks)

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
from the deployed API was unverified in this initial test. See the 100 ms retest
below, which did receive live partials. The adapter does not wait for partials.

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
- Broader language/voice coverage and statistically meaningful load benchmarks.

## 100 ms chunk retest

Historical comparison before automatic batching was added to the adapter.
At that revision, caller input sizes also controlled WebSocket message sizes.

The same Fast models, voice and short phrase were rerun three times per chunk
size, sequentially without concurrent test load. The test script now defaults
to 100 ms; 20 ms remains selectable. Both sizes use corrected capture pacing:
sleep before sending the captured chunk, then commit immediately after the
short tail. The initial script sent each chunk before its sleep, giving the
server an artificial head start; compare the fresh 20/100 ms results below
instead of attributing all differences from the initial results to chunk size.

| STT chunk | Commit to final, each run (ms) | Median (ms) | Runs with partials |
| --- | --- | ---: | ---: |
| 20 ms | 120.80, 119.91, 122.29 | 120.80 | 0/3 |
| 100 ms | 88.57, 90.91, 82.60 | 88.57 | 3/3 |

All final transcripts matched the short phrase. 100 ms produced partials in
all three runs; 20 ms produced none. This is an observed association in this
small sample, not an established server-side explanation or a p99 guarantee.
The first TTS call in each group used a new HTTP connection; later calls reused
it. STT chunk size does not change the TTS request or its output buffering.

Raw measurements: [chunk-size-results.json](chunk-size-results.json). The
LiveKit first-partial clock starts before simulated audio capture; Pipecat's
starts at the first audio frame processed (after the initial capture interval),
so those partial offsets should not be compared directly between frameworks.

A further 14.3-second utterance with 100 ms chunks also
returned a partial and the correct full final transcript. Commit to final was
93.14 ms (one run).


## Automatic 100 ms batching acceptance

The production adapter now sends 100 ms STT messages even when callers supply
20 ms audio frames. `scripts/smoke.py --chunk-ms 20 --runs 3` exercised the actual
adapter against the Fast API after this change. All three short-phrase runs
produced live partials and correct final transcripts.

Commit to final: 96.30, 91.47, 82.58 ms; median 91.47 ms.
These are small network-inclusive smoke samples, not a statistical benchmark.
Raw measurements: [automatic-batching-results.json](automatic-batching-results.json).

Local HTTP/WebSocket tests inspect actual outbound message bytes: five 20 ms
frames make one 3,200-byte append; a partial tail precedes commit without padding;
consecutive utterances have no loss, duplication or cross-turn tail mixing;
cancellation discards an unsent tail. The framework tests exercise explicit
VAD boundaries and graceful end. Original input frames remain unchanged.

The headless AgentSession/Silero test also passed again with 20 ms input:
`speaking -> partials -> listening -> final`. STT batching did not require changing
VAD input frames or manually injecting a speech-end event.
