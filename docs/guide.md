# LiveKit integration guide

[Back to README](../README.md)

## VAD and transcription

Create one recognizer per session. `bind()` listens to LiveKit's public
`user_state_changed` event and sends a Nari commit on `speaking -> listening`.
It **does not** commit the conversation turn. LiveKit's VAD, optional semantic
turn detector, endpointing, and interruption handling retain control. The
VAD model runs once, inside LiveKit; Nari server VAD is disabled by default.
You can replace `"vad"` with a LiveKit turn detector without changing the binding.
Do not use `turn_detection="stt"` with `bind()`.

For standalone transcription, use `recognizer.stream()`, `push_frame()`, and
`flush()` at each utterance boundary. `end_input()` commits the last segment
and waits for all outstanding final transcripts. `aclose()` cancels immediately.
Input must be mono; other input sample rates are resampled to 16 kHz by LiveKit.
As an explicit alternative, use `STT(turn_detection={"type": "server_vad"})`
without `bind()`, and keep sending silence between speech segments.

## Options

| Service | Options |
| --- | --- |
| Both | `api_key` (defaults to `NARI_API_KEY`), `model`, `base_url` |
| STT | `language=None`, `prompt=""`, `turn_detection=None`, `final_timeout=15` seconds |
| TTS | `voice="diana"`, `language=None`, `seed=0`, `read_timeout=30` seconds, optional `http_session`, `sentence_tokenizer` |

Fast models are the defaults. Standard IDs `qwen3-asr` / `qwen3-tts` are also
accepted. A TTS language must match the voice's assigned language; omission
uses that voice's language. STT settings are fixed for each WebSocket.
TTS outputs 24 kHz mono PCM16. The plugin omits unset optional fields and sends
only documented Nari request fields.

## Latency behavior

- STT input is accumulated into 100 ms (3,200-byte) PCM16 WebSocket messages.
  This happens inside the plugin regardless of the caller's input frame size.
  LiveKit VAD still receives the original frames.
- A VAD boundary sends the remaining short audio chunk before committing,
  without padding or waiting for another full chunk. There is no debounce timer.
- TTS starts each completed sentence while the LLM continues generating. The
  default tokenizer does not merge short sentences to meet a minimum length.
- The plugin implements its own LiveKit text stream, avoiding the generic
  StreamAdapter's second audio buffer. It uses 20 ms output frames and flushes
  available partial frames immediately, including the first HTTP chunk.
- One HTTP client is reused across TTS requests. PCM skips WAV/container decoding.
- Input over 2,048 characters is split at sentence/word boundaries. Synthesis
  requests are sequential to preserve order and avoid speculative wasted audio.
- Interruptions close in-flight responses and cancel queued synthesis.

Nari accepts complete text per HTTP request, not incremental tokens on one
request. Punctuation/LLM arrival time still bounds the first sentence's start.
VAD silence duration, semantic endpointing, network RTT, and playback buffering
also contribute to end-to-end latency. The example's 200 ms VAD silence is a
starting point, not an accuracy recommendation for every caller or language.
Preserve normal capitalization and punctuation for speech quality.

## Failure and transcript semantics

Partials replace the current hypothesis; they are not text deltas. Final results
include `item_id` and `commit_reason` in `SpeechData.metadata`. A 36-second
`max_duration` boundary yields a final segment without an end-of-speech event.
`commit_empty` never discards earlier pending finals.

Errors preserve provider codes/request IDs and map to LiveKit status, timeout,
or connection errors. Retries happen inside the adapter, before STT starts
consuming audio or before TTS receives HTTP 200. The limit is
`APIConnectOptions.max_retry` (LiveKit default: 3 retries); set it to zero to
disable. Backoff starts at 250 ms, doubles, and adds up to 100 ms jitter.
`Retry-After` is honored; waits over five seconds are surfaced to the application
instead of retrying early. Timeout settings apply per attempt. Exhausted failures
are not retried again by LiveKit, which would replay an entire stream.

Authentication, credits and invalid requests are not retried. A started TTS
response is never replayed, even if it fails before the first PCM chunk. Each
long-text/sentence request has its own retry budget, so earlier speech is not
repeated. STT disconnection or missing finals ends that stream with an error;
keep already received finals and explicitly start a new stream or use a fallback.
No incomplete utterance is silently resumed or replayed.
Batch recognition, diarization, and word timestamps are not exposed in this release.

Close the STT/TTS instances when the session ends; the [example](../examples/agent.py) registers a
shutdown callback. Externally supplied HTTP sessions remain caller-owned.

## Metrics and connection preparation

Subscribe to `recognizer.on("metrics_collected", handler)` for standard LiveKit
STT usage. The plugin emits `RECOGNITION_USAGE` increments on finals and stream
teardown, tagged with the connection request ID and model/provider metadata.
`audio_duration` measures PCM successfully written to the WebSocket, including
silence. It excludes locally buffered tails discarded on cancellation. These
increments are client transport observations, not proof of server acceptance or
invoice amounts. TTS retains LiveKit's standard metrics. The separate smoke
script measures commit-to-final latency; STT usage events do not supply that
latency measurement.

Call `synthesizer.prewarm()` for background preparation (used in the example),
or `await synthesizer.warmup()` to wait explicitly and receive any warmup error.
Preparation uses authenticated `GET /v1/voices?model=...` on the same HTTP pool;
it generates no speech. It has a five-second total timeout. Background errors
are logged, and later synthesis remains possible. Concurrent warmup calls share
the active task; closing the provider cancels it. An external HTTP session is
never closed by the plugin.

## Test and measure

```bash
uv run ruff check .
uv run pytest -q
uv build
# Uses paid Nari API calls; prints timings and the known test phrase, never the key:
uv run --extra examples python scripts/smoke.py --env-file .env --runs 3 --chunk-ms 20 --prewarm
# Headless AgentSession with real Silero VAD (also uses paid API calls):
uv run --extra examples python scripts/vad_smoke.py --env-file .env
```

The smoke test synthesizes a phrase, resamples it, streams it at microphone
speed, and checks for a nonempty final transcript. `tts_first_frame_ms` includes
connection establishment on the first run; later runs reuse the HTTP connection.
`stt_commit_to_final_ms` starts at the client's explicit commit, excluding VAD
silence and user speech. These are client observations, not server-only latency.
See [validation results](validation.md).

The API smoke script defaults to `--chunk-ms 100` (3,200 bytes of 16 kHz
mono PCM16). Use `--chunk-ms 20` to exercise automatic batching of smaller
inputs. Each chunk is sent after its simulated capture interval; the short final chunk is sent without padding and
committed immediately. This option controls the smoke test input frame size;
the adapter now always batches STT transmission into 100 ms chunks, flushing shorter tails on
commit. JSON reports the input size as `stt_chunk_ms` and the regular WebSocket
chunk size as `stt_wire_chunk_ms`. Cancelling/closing discards unsent tails;
graceful end commits them.
See the [chunk-size comparison](validation.md#100-ms-chunk-retest).

## Release workflow

CI runs lint, local HTTP/WebSocket tests, and package builds. Publishing is
deliberately not automated while the repository is private. Before a public
release, review package names/version bounds, run microphone and interruption
acceptance tests, publish the package, and then propose the upstream plugin.
