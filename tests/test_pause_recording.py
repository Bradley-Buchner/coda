import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from coda.app import server
from coda.dialogue import StreamingTranscriber, TranscriptEvent


class FakeReceiveWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    async def receive(self):
        return self.messages.pop(0)


class LaggingTranscriber(StreamingTranscriber):
    """Commits each frame's text only once the next frame arrives, like a
    streaming backend holding its last words until more audio comes."""

    def __init__(self, log):
        self.log = log

    async def stream(self, audio, *, language="en", task="transcribe"):
        held = None
        async for data in audio:
            if data == bytes(server.PAUSE_SILENCE_BYTES):
                self.log.append("silence")
            if held:
                yield TranscriptEvent(id=held, timestamp=0.0, text=held,
                                      committed=True)
            held = None if not data.strip(b"\0") else data.decode()
        if held:
            yield TranscriptEvent(id=held, timestamp=0.0, text=held,
                                  committed=True)


@pytest.mark.asyncio
async def test_capture_queues_control_messages_in_order():
    websocket = FakeReceiveWebSocket([
        {"type": "websocket.receive", "bytes": b"a"},
        {"type": "websocket.receive", "text": json.dumps({"type": "pause"})},
        {"type": "websocket.receive", "text": "not json"},
        {"type": "websocket.receive", "text": json.dumps({"type": "resume"})},
        {"type": "websocket.disconnect", "code": 1000},
    ])
    queue = asyncio.Queue()
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(websocket, queue)
    items = [queue.get_nowait() for _ in range(queue.qsize())]
    assert items == [b"a", "pause", "resume", None]


@pytest.mark.asyncio
async def test_no_inference_while_paused(monkeypatch):
    log = []

    async def fake_handle_committed(session, event, direct_translate):
        return event.id, event.timestamp, event.text, []

    async def fake_start_inference(session, chunk_id, timestamp, text, anns):
        log.append(text)

    monkeypatch.setattr(server, "transcriber", LaggingTranscriber(log))
    monkeypatch.setattr(server, "_handle_committed", fake_handle_committed)
    monkeypatch.setattr(server, "_start_inference", fake_start_inference)
    monkeypatch.setattr(server, "INFERENCE_MIN_WORDS", 2)

    queue = asyncio.Queue()
    for item in (b"fever for days", "pause", "resume", b"then cough", None):
        queue.put_nowait(item)
    session = server.InferenceSessionCoordinator(None, session_id="s")
    await server.consume_transcripts(None, queue, session)

    assert log == ["silence", "fever for days", "then cough"]
