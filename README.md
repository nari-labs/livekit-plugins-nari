# Nari Labs for LiveKit Agents

Streaming speech recognition and synthesis using the [Nari API](https://docs.narilabs.com).
Maintained by Nari Labs. Author: **shamuiscoding <toebee@snu.ac.kr>**.

Private development preview; not published to PyPI and not an upstream LiveKit plugin.
Python 3.11+; developed against LiveKit Agents 1.8.2. No GPU is required by this plugin.

## Install from this checkout

```bash
uv sync --extra examples
cp .env.example .env
```

Set your Nari key, OpenRouter key and an available OpenRouter model ID in `.env`.
The example uses the English voice `diana`; choose a voice with the correct
language for multilingual agents. Voice IDs are model-specific.

```bash
uv run --extra examples python examples/agent.py console
```

For a LiveKit room, also configure `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and
`LIVEKIT_API_SECRET`, then run `examples/agent.py dev`. The example uses the
existing OpenRouter LLM integration; no Nari model listing on OpenRouter is required.

## Use in an existing agent

```python
from livekit.agents import AgentSession
from livekit.plugins import nari, silero

recognizer = nari.STT(model="qwen3-asr-fast")
session = AgentSession(
    stt=recognizer,
    tts=nari.TTS(model="qwen3-tts-fast", voice="diana"),
    vad=silero.VAD.load(min_silence_duration=0.2),
    llm=your_llm,
    turn_handling={"turn_detection": "vad"},
)
recognizer.bind(session)  # Before session.start(). Required for local VAD commits.
```

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

- Audio is forwarded continuously to STT without waiting for a full utterance.
- A VAD boundary queues a commit immediately; there is no plugin debounce timer.
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

Connection/configuration errors, missing finals, or incomplete PCM streams
surface as LiveKit API errors. This preview does not automatically retry or
replay failed requests: an interrupted utterance cannot resume, and retrying
partially played TTS would repeat speech. Configure application-level recovery
or a fallback provider. Error messages preserve Nari request IDs when available.
Batch recognition, diarization, and word timestamps are not exposed in this release.

Close the STT/TTS instances when the session ends; the example registers a
shutdown callback. Externally supplied HTTP sessions remain caller-owned.

## Test and measure

```bash
uv run ruff check .
uv run pytest -q
uv build
# Uses paid Nari API calls; prints timings and the known test phrase, never the key:
uv run --extra examples python scripts/smoke.py --env-file .env --runs 3 --chunk-ms 100
# Headless AgentSession with real Silero VAD (also uses paid API calls):
uv run --extra examples python scripts/vad_smoke.py --env-file .env
```

The smoke test synthesizes a phrase, resamples it, streams it at microphone
speed, and checks for a nonempty final transcript. `tts_first_frame_ms` includes
connection establishment on the first run; later runs reuse the HTTP connection.
`stt_commit_to_final_ms` starts at the client's explicit commit, excluding VAD
silence and user speech. These are client observations, not server-only latency.
See [validation results](docs/validation.md).

## Release workflow

CI runs lint, local HTTP/WebSocket tests, and package builds. Publishing is
deliberately not automated while the repository is private. Before a public
release, review package names/version bounds, run microphone and interruption
acceptance tests, publish the package, and then propose the upstream plugin.

The API smoke script defaults to `--chunk-ms 100` (3,200 bytes of 16 kHz
mono PCM16). Use `--chunk-ms 20` for comparison. Each chunk is sent after its
simulated capture interval; the short final chunk is sent without padding and
committed immediately. This option changes the smoke test input, not the adapter's
production buffering: the adapter continues forwarding frames supplied by its caller.
See the [chunk-size comparison](docs/validation.md#100-ms-chunk-retest).
