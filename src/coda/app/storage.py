"""Per-case storage of interview data.

Each case gets its own folder under ``storage.output_dir`` holding a
``manifest.json`` plus one file per enabled ``storage.store`` item. A
``CaseRecorder`` is only created when ``storage.enabled`` is true.
"""

import json
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import coda
from coda import CODA_BASE
from coda.config import settings

AUDIO_SAMPLE_RATE = 16000


def storage_enabled() -> bool:
    return bool(settings.storage.get("enabled", False))


def cases_root() -> Path:
    output_dir = settings.storage.get("output_dir", "")
    if output_dir:
        return Path(output_dir).expanduser()
    return CODA_BASE.join(name="cases")


class CaseRecorder:
    """Writes everything recorded for one case into its own folder."""

    def __init__(self, session_id: str, generation: int, run_info: dict):
        self.store = {key: bool(value)
                      for key, value in settings.storage.store.items()}
        started = datetime.now()
        self.case_id = (f"{started.strftime('%Y-%m-%d_%H%M%S')}"
                        f"_{session_id[:8]}_g{generation}")
        self.case_dir = cases_root() / self.case_id
        self.case_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            "case_id": self.case_id,
            "session_id": session_id,
            "generation": generation,
            "started_at": started.isoformat(),
            "ended_at": None,
            "pauses": [],
            "coda_version": coda.__version__,
            "run_info": run_info,
        }
        self._write_manifest()

        self._transcripts: dict[str, TextIO] = {}
        self._chunks = self._open("chunks.jsonl", "annotations")
        self._inference = self._open("inference.jsonl", "inference")
        self._timing = self._open("timing.jsonl", "timing")
        self._audio: Optional[wave.Wave_write] = None
        if self.store.get("audio"):
            self._audio = wave.open(str(self.case_dir / "audio.wav"), "wb")
            self._audio.setnchannels(1)
            self._audio.setsampwidth(2)
            self._audio.setframerate(AUDIO_SAMPLE_RATE)

    def _open(self, filename: str, store_key: str) -> Optional[TextIO]:
        if not self.store.get(store_key):
            return None
        return open(self.case_dir / filename, "a", encoding="utf-8")

    def _write_manifest(self):
        with open(self.case_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    @staticmethod
    def _append_json(f: Optional[TextIO], record: dict):
        if f is None:
            return
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

    def _append_transcript(self, text: str, lang_code: str):
        f = self._transcripts.get(lang_code)
        if f is None:
            f = open(self.case_dir / f"transcript_{lang_code}.txt", "a",
                     encoding="utf-8")
            self._transcripts[lang_code] = f
        f.write(text + "\n")
        f.flush()

    def write_audio(self, data: bytes):
        if self._audio is not None:
            self._audio.writeframes(data)

    def write_pause(self):
        self.manifest["pauses"].append(
            {"paused_at": datetime.now().isoformat(), "resumed_at": None})
        self._write_manifest()

    def write_resume(self):
        pauses = self.manifest["pauses"]
        if pauses and pauses[-1]["resumed_at"] is None:
            pauses[-1]["resumed_at"] = datetime.now().isoformat()
            self._write_manifest()

    def write_chunk(self, chunk_id: str, timestamp: float, english_text: str,
                    annotations: list, timings: dict,
                    original_text: Optional[str] = None,
                    original_language: Optional[str] = None):
        received_at = time.time()
        if self.store.get("transcripts"):
            self._append_transcript(english_text, "en")
            if original_text and original_language:
                self._append_transcript(original_text, original_language)

        record: dict[str, Any] = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "received_at": received_at,
            "text": english_text,
            "annotations": [a.to_json() for a in annotations] if annotations else [],
        }
        if original_text:
            record["original_text"] = original_text
            record["original_language"] = original_language
        self._append_json(self._chunks, record)
        self._append_json(self._timing, {
            "kind": "chunk", "chunk_id": chunk_id, "at": received_at, **timings,
        })

    def write_inference(self, request: dict, result: dict, shown_at: float):
        record = {
            "shown_at": shown_at,
            "chunk_id": result.get("chunk_id", request.get("chunk_id")),
            "timestamp": result.get("timestamp", request.get("timestamp")),
            "chunks_processed": result.get("chunks_processed"),
            "causes": result.get("causes"),
            "reasoning": result.get("reasoning"),
            "questions": result.get("questions"),
        }
        if self.store.get("metadata"):
            record["metadata"] = request.get("metadata")
        self._append_json(self._inference, record)
        self._append_json(self._timing, {
            "kind": "inference", "chunk_id": record["chunk_id"],
            "at": shown_at, **(result.get("timings") or {}),
        })

    def close(self, metadata: Optional[dict] = None):
        for f in [*self._transcripts.values(), self._chunks, self._inference,
                  self._timing]:
            if f is not None:
                f.close()
        self._transcripts.clear()
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self.manifest["ended_at"] = datetime.now().isoformat()
        if self.store.get("metadata"):
            self.manifest["metadata"] = metadata
        self._write_manifest()
