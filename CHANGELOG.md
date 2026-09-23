# Changelog

## 0.1.0 (unreleased)

- Batch STT transmission into 100 ms messages automatically; flush short tails
  before each commit and discard them on cancellation. VAD frame sizes stay unchanged.

- Add Nari realtime STT with LiveKit VAD/session binding.
- Add sentence-streamed TTS with immediate PCM forwarding and cancellation.
- Add local protocol tests, OpenRouter voice example and live latency smoke test.
