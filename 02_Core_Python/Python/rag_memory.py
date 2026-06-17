"""
Local RAG memory for Chain Gambler.

Stores lightweight text chunks and Ollama embeddings in JSONL so the Windows VPS
can build searchable memory without a database. This is intentionally simple:
small enough to debug, good enough for trade logs, reports, configs, and notes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from Python.ollama_advisor import OllamaClient, OllamaSettings, load_ollama_settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "memory" / "ollama_memory.jsonl"
DEFAULT_INDEX_DIRS = [ROOT / "logs", ROOT / "docs" / "results", ROOT / "configs"]
TEXT_SUFFIXES = {".txt", ".log", ".json", ".jsonl", ".yaml", ".yml", ".md", ".csv"}


@dataclass
class MemoryRecord:
    record_id: str
    source: str
    chunk_index: int
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "record_id": self.record_id,
                "source": self.source,
                "chunk_index": self.chunk_index,
                "text": self.text,
                "embedding": self.embedding,
                "metadata": self.metadata,
                "created_at": self.created_at,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=str(data["record_id"]),
            source=str(data["source"]),
            chunk_index=int(data.get("chunk_index", 0)),
            text=str(data.get("text", "")),
            embedding=[float(x) for x in data.get("embedding", [])],
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", 0.0)),
        )


class LocalRagMemory:
    """Tiny vector memory backed by a JSONL file."""

    def __init__(
        self,
        store_path: Union[Path, str] = DEFAULT_STORE,
        client: Optional[OllamaClient] = None,
        settings: Optional[OllamaSettings] = None,
    ):
        self.store_path = Path(store_path)
        self.settings = settings or load_ollama_settings()
        self.client = client or OllamaClient(self.settings)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def add_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_chars: int = 1800,
        overlap: int = 200,
    ) -> List[MemoryRecord]:
        """Embed and store text chunks."""
        records: List[MemoryRecord] = []
        existing_ids = self._existing_ids()
        for idx, chunk in enumerate(chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)):
            record_id = make_record_id(source, idx, chunk)
            if record_id in existing_ids:
                continue
            embedding = self.client.embed(text=chunk)
            record = MemoryRecord(
                record_id=record_id,
                source=source,
                chunk_index=idx,
                text=chunk,
                embedding=embedding,
                metadata=metadata or {},
                created_at=time.time(),
            )
            records.append(record)

        if records:
            with self.store_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(record.to_json() + "\n")
        return records

    def index_file(self, path: Union[Path, str]) -> List[MemoryRecord]:
        """Index one text-like file."""
        path = Path(path)
        if not path.exists() or not path.is_file():
            return []
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning(f"Could not read memory file {path}: {exc}")
            return []
        metadata = {"path": str(path), "suffix": path.suffix, "size_bytes": path.stat().st_size}
        try:
            source = str(path.relative_to(ROOT))
        except ValueError:
            source = str(path)
        return self.add_text(text=text, source=source, metadata=metadata)

    def index_paths(self, paths: Iterable[Union[Path, str]]) -> Dict[str, Any]:
        """Index files and directories."""
        files = list(iter_text_files(paths))
        total_records = 0
        indexed_files = 0
        errors: List[str] = []
        for path in files:
            try:
                added = self.index_file(path)
                if added:
                    indexed_files += 1
                    total_records += len(added)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return {"files_seen": len(files), "files_indexed": indexed_files, "records_added": total_records, "errors": errors}

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Return top memory chunks by cosine similarity."""
        records = self.load_records()
        if not records:
            return []
        query_vec = self.client.embed(text=query)
        scored: List[Tuple[float, MemoryRecord]] = []
        for record in records:
            score = cosine_similarity(query_vec, record.embedding)
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": round(score, 6),
                "source": record.source,
                "chunk_index": record.chunk_index,
                "text": record.text,
                "metadata": record.metadata,
            }
            for score, record in scored[:top_k]
        ]

    def load_records(self) -> List[MemoryRecord]:
        if not self.store_path.exists():
            return []
        records: List[MemoryRecord] = []
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(MemoryRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    logger.debug(f"Skipping bad memory record: {exc}")
        return records

    def _existing_ids(self) -> Set[str]:
        return {record.record_id for record in self.load_records()}


def iter_text_files(paths: Iterable[Union[Path, str]]) -> Iterable[Path]:
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                yield child


def chunk_text(text: str, chunk_chars: int = 1800, overlap: int = 200) -> List[str]:
    clean = "\n".join(line.rstrip() for line in text.splitlines())
    if not clean.strip():
        return []
    if len(clean) <= chunk_chars:
        return [clean]

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_chars, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def make_record_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text}".encode("utf-8", errors="replace")).hexdigest()
    return digest[:32]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chain Gambler local Ollama RAG memory")
    parser.add_argument("command", choices=["index", "search", "stats"])
    parser.add_argument("query", nargs="?", help="Search query when command=search")
    parser.add_argument("--path", action="append", help="Path to index. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    args = parser.parse_args()

    memory = LocalRagMemory(store_path=args.store)

    if args.command == "index":
        paths = [Path(p) for p in args.path] if args.path else DEFAULT_INDEX_DIRS
        print(json.dumps(memory.index_paths(paths), indent=2))
        return 0

    if args.command == "search":
        if not args.query:
            parser.error("search requires a query")
        print(json.dumps(memory.search(args.query, top_k=args.top_k), indent=2))
        return 0

    records = memory.load_records()
    print(json.dumps({"store": str(memory.store_path), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
