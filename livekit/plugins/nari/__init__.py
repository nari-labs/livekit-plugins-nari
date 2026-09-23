"""Nari Labs STT and TTS for LiveKit Agents."""

import logging

from livekit.agents import Plugin

from .stt import STT
from .tts import TTS

__version__ = "0.1.0"
__all__ = ["STT", "TTS", "__version__"]


class NariPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logging.getLogger(__name__))


Plugin.register_plugin(NariPlugin())
