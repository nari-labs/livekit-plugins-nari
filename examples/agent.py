"""Local console or LiveKit room voice agent with Nari speech and OpenRouter."""

import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli

from livekit.plugins import nari, openai, silero

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    recognizer = nari.STT(model=os.getenv("NARI_STT_MODEL", "qwen3-asr-fast"))
    synthesizer = nari.TTS(
        model=os.getenv("NARI_TTS_MODEL", "qwen3-tts-fast"),
        voice=os.getenv("NARI_VOICE", "diana"),
    )
    synthesizer.prewarm()
    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=0.2),
        stt=recognizer,
        tts=synthesizer,
        llm=openai.LLM.with_openrouter(model=os.environ["OPENROUTER_MODEL"]),
        # Start with local VAD. A LiveKit semantic turn detector can replace
        # "vad" without changing the Nari binding below.
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": 0.2, "max_delay": 3.0},
            "interruption": {"enabled": True},
        },
    )
    recognizer.bind(session)

    async def cleanup():
        await recognizer.aclose()
        await synthesizer.aclose()

    ctx.add_shutdown_callback(cleanup)
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "You are a helpful voice assistant. Answer briefly in English. "
                "Use normal capitalization and punctuation, with short complete sentences."
            )
        ),
    )
    await session.generate_reply(instructions="Greet the user briefly.")


if __name__ == "__main__":
    cli.run_app(server)
