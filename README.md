# Nari plugin for LiveKit Agents

Streaming speech-to-text and text-to-speech with the [Nari API](https://docs.narilabs.com).
Maintained by Nari Labs. Python 3.11+ · Tested with LiveKit Agents 1.8.2.

## Installation

Private preview — install from this checkout:

```bash
uv sync --extra examples
cp .env.example .env
```

Set `NARI_API_KEY` in `.env`. The example also needs `OPENROUTER_API_KEY` and
`OPENROUTER_MODEL`.

## Usage

Add Nari to an existing agent with your LLM:

```python
from livekit.agents import AgentSession
from livekit.plugins import nari, silero

recognizer = nari.STT()
session = AgentSession(
    stt=recognizer,
    tts=nari.TTS(voice="diana"),
    llm=your_llm,
    vad=silero.VAD.load(min_silence_duration=0.2),
    turn_handling={"turn_detection": "vad"},
)
recognizer.bind(session)  # Required before session.start().
```

Defaults: `qwen3-asr-fast` and `qwen3-tts-fast`. `diana` is an English voice.
Create one recognizer per session; LiveKit handles VAD, turns and interruptions.
Custom applications can pass `api_key=` or export `NARI_API_KEY`.

## Run the example

The [voice agent](examples/agent.py) connects Nari STT → OpenRouter → Nari TTS:

```bash
uv run --extra examples python examples/agent.py console
```

For room mode, set the `LIVEKIT_*` credentials in `.env` and replace `console`
with `dev`.

## Documentation

- [Configuration, streaming, metrics and recovery](docs/guide.md)
- [Tests and measured results](docs/validation.md)
- [Changelog](CHANGELOG.md) · [MIT license](LICENSE)
