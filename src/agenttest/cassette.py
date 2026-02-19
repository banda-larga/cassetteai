from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CASSETTE_VERSION = 1


def _hash_request(messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
    """Stable hash of (messages, tools) — the cache key."""
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class CassetteEntry:
    """One recorded LLM request/response pair."""

    request_hash: str
    request: dict[str, Any]  # full request body
    response: dict[str, Any]  # full response body (non-streaming)
    # For streaming: list of SSE data payloads (each is a dict)
    stream_chunks: list[dict[str, Any]] = field(default_factory=list)
    is_streaming: bool = False
    # Cost tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    call_index: int = 0  # which call this was in the session (for display)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "request": self.request,
            "response": self.response,
            "stream_chunks": self.stream_chunks,
            "is_streaming": self.is_streaming,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "call_index": self.call_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CassetteEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Cassette:
    """Ordered collection of CassetteEntry objects with save/load and lookup."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: list[CassetteEntry] = []
        self._lookup: dict[str, list[CassetteEntry]] = {}
        self._serve_counts: dict[str, int] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> list[CassetteEntry]:
        return list(self._entries)

    def add(self, entry: CassetteEntry) -> None:
        self._entries.append(entry)
        self._lookup.setdefault(entry.request_hash, []).append(entry)

    def match(
        self, messages: list[dict], tools: list[dict] | None
    ) -> CassetteEntry | None:
        """Return the next unserved entry for this request, or None on miss."""
        h = _hash_request(messages, tools)
        candidates = self._lookup.get(h, [])
        served = self._serve_counts.get(h, 0)
        if served < len(candidates):
            self._serve_counts[h] = served + 1
            return candidates[served]
        return None

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CASSETTE_VERSION,
            "entries": [e.to_dict() for e in self._entries],
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Cassette saved: %s (%d entries)", self._path, len(self._entries))

    @classmethod
    def load(cls, path: Path) -> Cassette:
        cassette = cls(path)
        if not path.exists():
            return cassette
        data = json.loads(path.read_text())
        if data.get("version") != CASSETTE_VERSION:
            raise ValueError(
                f"Cassette version mismatch: expected {CASSETTE_VERSION}, "
                f"got {data.get('version')}. Re-record this cassette."
            )
        for d in data.get("entries", []):
            cassette.add(CassetteEntry.from_dict(d))
        logger.info("Cassette loaded: %s (%d entries)", path, len(cassette._entries))
        return cassette

    def exists(self) -> bool:
        return self._path.exists()

    def __len__(self) -> int:
        return len(self._entries)
