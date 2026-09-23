# Changelog

## 0.1.0 (unreleased)

- Emit STT usage from transmitted PCM; connect framework metrics without counting
  unsent tails or duplicating usage at repeated commits.
- Prepare TTS HTTP connections through the voices endpoint without synthesis.
- Retry transient setup/rejected-request failures with bounded jittered backoff;
  classify permanent failures and prevent replay after streaming has begun.

- Batch STT transmission into 100 ms messages automatically; flush short tails
  before each commit and discard them on cancellation. VAD frame sizes stay unchanged.

- Add Nari realtime STT with LiveKit VAD/session binding.
- Add sentence-streamed TTS with immediate PCM forwarding and cancellation.
- Add local protocol tests, OpenRouter voice example and live latency smoke test.
