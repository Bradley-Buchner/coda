"""Tests for per-case storage (coda.app.storage) and its server wiring."""

import json
import logging
import wave
from types import SimpleNamespace

import pytest

from coda.app import server
from coda.app.storage import CaseRecorder
from coda.config import configure_logging, reload_settings

STORAGE_ENV_VARS = (
    "CODA_STORAGE__ENABLED",
    "CODA_STORAGE__OUTPUT_DIR",
    "CODA_STORAGE__STORE__AUDIO",
    "CODA_STORAGE__STORE__INFERENCE",
    "CODA_LOGGING__FILE",
)


class DummyWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        self.messages.append(data)


class FakeAnnotation:
    def __init__(self, text: str):
        self.text = text

    def to_json(self) -> dict:
        return {"text": self.text}


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self) -> dict:
        return dict(self.payload)


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    for var in STORAGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reload_settings()
    yield
    monkeypatch.undo()
    reload_settings()


@pytest.fixture
def storage_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CODA_STORAGE__ENABLED", "true")
    monkeypatch.setenv("CODA_STORAGE__OUTPUT_DIR", str(tmp_path))
    reload_settings()
    return tmp_path


@pytest.fixture
def offline_server(monkeypatch):
    async def fake_run_info() -> dict:
        return {"language": "en"}

    async def fake_reset(session_id, generation):
        return None

    monkeypatch.setattr(server, "_run_info", fake_run_info)
    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)
    monkeypatch.setattr(server, "grounder", SimpleNamespace(annotate=lambda text: []))
    monkeypatch.setattr(server, "current_language", "en")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def case_dirs(root) -> list:
    return sorted(p for p in root.iterdir() if p.is_dir())


def test_recorder_writes_case_files(storage_dir):
    recorder = CaseRecorder("session-abcdef12", 0, {"language": "sw"})
    recorder.write_chunk("c1", 1.5, "fever for three days",
                         [FakeAnnotation("fever")], {"translate_s": 0.1},
                         original_text="homa kwa siku tatu",
                         original_language="sw")
    recorder.write_inference(
        {"chunk_id": "c1", "timestamp": 1.5, "metadata": {"age": 3}},
        {"causes": {"X": {"score": 0.9}}, "questions": ["How old?"],
         "timings": {"request_s": 1.0}},
        shown_at=100.0,
    )
    recorder.close({"age": 3})

    case_dir = recorder.case_dir
    assert case_dir.parent == storage_dir
    assert case_dir.name.endswith("_session-_g0")
    assert (case_dir / "transcript_en.txt").read_text() == "fever for three days\n"
    assert (case_dir / "transcript_sw.txt").read_text() == "homa kwa siku tatu\n"

    [chunk] = read_jsonl(case_dir / "chunks.jsonl")
    assert chunk["annotations"] == [{"text": "fever"}]
    assert chunk["original_language"] == "sw"

    [inference] = read_jsonl(case_dir / "inference.jsonl")
    assert inference["questions"] == ["How old?"]
    assert inference["shown_at"] == 100.0
    assert inference["metadata"] == {"age": 3}

    assert [t["kind"] for t in read_jsonl(case_dir / "timing.jsonl")] == \
        ["chunk", "inference"]

    manifest = json.loads((case_dir / "manifest.json").read_text())
    assert manifest["run_info"] == {"language": "sw"}
    assert manifest["ended_at"] is not None
    assert manifest["metadata"] == {"age": 3}
    assert not (case_dir / "audio.wav").exists()


def test_store_flags_disable_files(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__INFERENCE", "false")
    reload_settings()
    recorder = CaseRecorder("session-a", 0, {})
    recorder.write_inference({}, {"questions": ["q"]}, shown_at=1.0)
    recorder.close()
    assert not (recorder.case_dir / "inference.jsonl").exists()


def test_audio_is_written_as_wav(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    recorder = CaseRecorder("session-a", 0, {})
    recorder.write_audio(b"\x00\x01" * 1600)
    recorder.close()
    with wave.open(str(recorder.case_dir / "audio.wav")) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 1600


@pytest.mark.asyncio
async def test_nothing_written_when_storage_disabled(offline_server, tmp_path,
                                                     monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__OUTPUT_DIR", str(tmp_path))
    reload_settings()
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()
    event = SimpleNamespace(id="c1", timestamp=0.0, text="fever")
    await server._handle_committed(session, event, direct_translate=False)
    await session.invalidate("reset")
    await session.invalidate("disconnect")

    assert session.recorder is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_reset_starts_new_case(offline_server, storage_dir):
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()
    event = SimpleNamespace(id="c1", timestamp=0.0, text="fever")
    await server._handle_committed(session, event, direct_translate=False)
    await session.invalidate("reset")
    await session.invalidate("disconnect")

    assert session.recorder is None
    first, second = case_dirs(storage_dir)
    assert first.name.endswith("_g0") and second.name.endswith("_g1")
    assert len(read_jsonl(first / "chunks.jsonl")) == 1
    assert (second / "chunks.jsonl").read_text() == ""
    for case_dir in (first, second):
        manifest = json.loads((case_dir / "manifest.json").read_text())
        assert manifest["ended_at"] is not None


@pytest.mark.asyncio
async def test_targeted_reset_only_rotates_target_case(offline_server,
                                                       storage_dir):
    session_a = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    session_b = server.InferenceSessionCoordinator(DummyWebSocket(), "session-b")
    await session_a.start_case()
    await session_b.start_case()
    recorder_b = session_b.recorder
    original_sessions = set(server.active_inference_sessions)
    server.active_inference_sessions.clear()
    server.active_inference_sessions.update({session_a, session_b})
    try:
        await server.reset_session(server.ResetRequest(session_id="session-a"))
    finally:
        server.active_inference_sessions.clear()
        server.active_inference_sessions.update(original_sessions)

    assert session_a.recorder.case_id.endswith("_g1")
    assert session_b.recorder is recorder_b
    session_a.end_case()
    session_b.end_case()


@pytest.mark.asyncio
async def test_inference_results_recorded_unless_stale(offline_server,
                                                       storage_dir, monkeypatch):
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()

    async def fake_post(path, json):
        return FakeResponse({"chunk_id": json["chunk_id"], "questions": ["q1"]})

    monkeypatch.setattr(server, "inference_client",
                        SimpleNamespace(post=fake_post))
    batch = server.CoalescedInferenceBatch(
        chunk_id="c1", timestamp=0.0, text="fever", annotations=[],
        queued_at=0.0, chunk_count=1)
    await server.process_inference(session, 0, batch, buffer_wait_s=0.0)
    await server.process_inference(session, 7, batch, buffer_wait_s=0.0)
    case_dir = session.recorder.case_dir
    session.end_case()

    [record] = read_jsonl(case_dir / "inference.jsonl")
    assert record["questions"] == ["q1"]
    assert record["shown_at"] is not None


def test_configure_logging_writes_to_file(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "coda.log"
    monkeypatch.setenv("CODA_LOGGING__FILE", str(log_file))
    reload_settings()
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    try:
        configure_logging()
        logging.getLogger("coda.test").info("hello storage")
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in root.handlers:
            if handler not in previous_handlers:
                handler.close()
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
    assert "hello storage" in log_file.read_text()
