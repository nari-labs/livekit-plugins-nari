"""Local HTTP/WebSocket peer implementing the public Nari wire contract."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web


class FakeNari:
    def __init__(self):
        self.requests = []
        self.events = asyncio.Queue()
        self.connections = 0
        self.socket = None
        self.pcm = b"\x01\x00" * 2400
        self.audio_gate = None
        self.final_gate = None
        self.tts_status = 200
        self.tts_complete = False
        self.tts_calls = 0
        self.tasks = []

    async def speech(self, request):
        self.tts_calls += 1
        self.requests.append((dict(request.headers), await request.json()))
        if self.tts_status != 200:
            return web.json_response(
                {
                    "error": {
                        "code": "INSUFFICIENT_CREDITS",
                        "message": "Add credits",
                        "requestId": "request-test",
                    }
                },
                status=self.tts_status,
            )
        response = web.StreamResponse(
            headers={"Content-Type": "audio/pcm", "x-request-id": "request-test"}
        )
        await response.prepare(request)
        # Split an int16 sample at an HTTP read boundary deliberately.
        await response.write(self.pcm[:961])
        if self.audio_gate:
            await self.audio_gate.wait()
        await response.write(self.pcm[961:])
        await response.write_eof()
        self.tts_complete = True
        return response

    async def realtime(self, request):
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.socket = ws
        count = 0
        item = 0
        async for message in ws:
            event = json.loads(message.data)
            self.requests.append((dict(request.headers), event))
            await self.events.put(event)
            if event["type"] == "session.configure":
                await ws.send_json({"type": "session.configured", "session": event["session"]})
            elif event["type"] == "input_audio_buffer.append":
                count += len(base64.b64decode(event["audio"]))
                for text in ["I scream", "Ice cream"]:
                    await ws.send_json(
                        {
                            "type": "transcript.partial",
                            "item_id": f"item_{item}",
                            "transcript": text,
                        }
                    )
            elif event["type"] == "input_audio_buffer.commit":
                if not count:
                    await ws.send_json(
                        {
                            "type": "input_audio_buffer.commit_empty",
                            "client_event_id": event["event_id"],
                            "item_id": None,
                        }
                    )
                    continue
                item_id = f"item_{item}"
                item += 1
                count = 0
                await ws.send_json(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": item_id,
                        "client_event_id": event["event_id"],
                    }
                )
                self.tasks.append(asyncio.create_task(self.complete(ws, item_id)))
        return ws

    async def complete(self, ws, item_id):
        if self.final_gate:
            await self.final_gate.wait()
        if not ws.closed:
            await ws.send_json(
                {
                    "type": "transcript.completed",
                    "item_id": item_id,
                    "transcript": "Ice cream is delicious.",
                    "language": "en",
                    "commit_reason": "manual",
                    "usage": {"input_audio_seconds": 0.02},
                }
            )

    async def next_event(self, kind):
        async with asyncio.timeout(2):
            while True:
                event = await self.events.get()
                if event["type"] == kind:
                    return event


@pytest.fixture
def server():
    @asynccontextmanager
    async def run():
        peer = FakeNari()
        app = web.Application()
        app.router.add_post("/v1/audio/speech", peer.speech)
        app.router.add_get("/v1/realtime", peer.realtime)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        peer.http_url = f"http://127.0.0.1:{port}/v1"
        peer.ws_url = f"ws://127.0.0.1:{port}/v1/realtime?intent=transcription"
        try:
            yield peer
        finally:
            if peer.socket:
                await peer.socket.close()
            for task in peer.tasks:
                task.cancel()
            await asyncio.gather(*peer.tasks, return_exceptions=True)
            await runner.cleanup()

    return run
