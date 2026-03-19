from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from neurai.models import Episode, Memory, MemorySearchResult, WorkingMemoryBlock
from neurai.storage.base import StorageBackend


def _normalize_embedding(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("embedding must be a non-empty 1D vector")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("embedding norm must be greater than zero")
    return array / norm


def _serialize_embedding(values: list[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float64).tolist()


def _coerce_timestamp(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _resolve_db_path(path: str) -> str:
    if path == ":memory:":
        return path
    if path.startswith("sqlite:///"):
        return path.removeprefix("sqlite:///")
    return path


class SQLiteStorage(StorageBackend):
    def __init__(self, path: str = "neurai.db") -> None:
        self.path = _resolve_db_path(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_current
                    ON memories(user_id, valid_until, valid_from DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_user_created
                    ON memories(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_topics TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_user_occurred
                    ON episodes(user_id, occurred_at DESC);

                CREATE TABLE IF NOT EXISTS working_memory (
                    user_id TEXT NOT NULL,
                    block_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, block_name)
                );

                CREATE INDEX IF NOT EXISTS idx_working_memory_user_updated
                    ON working_memory(user_id, updated_at DESC);
                """
            )
            self._conn.commit()

    def add_memory(self, memory: Memory) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, content, embedding, valid_from,
                    valid_until, created_at, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.content,
                    _serialize_embedding(memory.embedding),
                    memory.valid_from.isoformat(),
                    memory.valid_until.isoformat() if memory.valid_until else None,
                    memory.created_at.isoformat(),
                    memory.embedding_model,
                ),
            )
            self._conn.commit()

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._memory_from_row(row) if row else None

    def get_memories(
        self, user_id: str, include_retired: bool = False
    ) -> list[Memory]:
        query = "SELECT * FROM memories WHERE user_id = ?"
        params: list[object] = [user_id]
        if not include_retired:
            query += " AND valid_until IS NULL"
        query += " ORDER BY valid_from DESC, created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def retire_memory(self, memory_id: str, valid_until: datetime) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET valid_until = ? WHERE id = ?",
                (valid_until.isoformat(), memory_id),
            )
            self._conn.commit()

    def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        include_retired: bool = False,
    ) -> list[MemorySearchResult]:
        memories = self.get_memories(user_id, include_retired=include_retired)
        return self._rank_memories(memories, query_embedding, limit)

    def add_episode(self, episode: Episode) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO episodes (
                    id, user_id, summary, key_topics, embedding,
                    occurred_at, created_at, message_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.user_id,
                    episode.summary,
                    json.dumps(episode.key_topics),
                    _serialize_embedding(episode.embedding),
                    episode.occurred_at.isoformat(),
                    episode.created_at.isoformat(),
                    episode.message_count,
                ),
            )
            self._conn.commit()

    def get_episodes(
        self,
        user_id: str,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Episode]:
        query = "SELECT * FROM episodes WHERE user_id = ?"
        params: list[object] = [user_id]
        if after is not None:
            query += " AND occurred_at >= ?"
            params.append(after.isoformat())
        if before is not None:
            query += " AND occurred_at <= ?"
            params.append(before.isoformat())
        query += " ORDER BY occurred_at DESC, created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._episode_from_row(row) for row in rows]

    def search_episodes(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[tuple[Episode, float]]:
        episodes = self.get_episodes(user_id, after=after, before=before)
        query_vector = _normalize_embedding(query_embedding)
        ranked: list[tuple[Episode, float]] = []
        for episode in episodes:
            score = float(
                np.dot(query_vector, _normalize_embedding(episode.embedding))
            )
            ranked.append((episode, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def upsert_working_memory(self, block: WorkingMemoryBlock) -> None:
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT 1 FROM working_memory
                WHERE user_id = ? AND block_name = ?
                """,
                (block.user_id, block.block_name),
            ).fetchone()
            if existing is None:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM working_memory WHERE user_id = ?",
                    (block.user_id,),
                ).fetchone()[0]
                if count >= 10:
                    raise ValueError("working memory is limited to 10 blocks per user")

            self._conn.execute(
                """
                INSERT INTO working_memory (user_id, block_name, content, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, block_name) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (
                    block.user_id,
                    block.block_name,
                    block.content,
                    block.updated_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_working_memory(self, user_id: str) -> list[WorkingMemoryBlock]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM working_memory
                WHERE user_id = ?
                ORDER BY updated_at DESC, block_name ASC
                """,
                (user_id,),
            ).fetchall()
        return [self._working_memory_from_row(row) for row in rows]

    def delete_working_memory(self, user_id: str, block_name: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                DELETE FROM working_memory
                WHERE user_id = ? AND block_name = ?
                """,
                (user_id, block_name),
            )
            self._conn.commit()

    def delete_user_data(self, user_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
            self._conn.execute(
                "DELETE FROM working_memory WHERE user_id = ?",
                (user_id,),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _memory_from_row(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            embedding=_deserialize_embedding(row["embedding"]),
            valid_from=_coerce_timestamp(row["valid_from"]),
            valid_until=(
                _coerce_timestamp(row["valid_until"])
                if row["valid_until"] is not None
                else None
            ),
            created_at=_coerce_timestamp(row["created_at"]),
            embedding_model=row["embedding_model"],
        )

    def _episode_from_row(self, row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            summary=row["summary"],
            key_topics=json.loads(row["key_topics"]),
            embedding=_deserialize_embedding(row["embedding"]),
            occurred_at=_coerce_timestamp(row["occurred_at"]),
            created_at=_coerce_timestamp(row["created_at"]),
            message_count=row["message_count"],
        )

    def _working_memory_from_row(self, row: sqlite3.Row) -> WorkingMemoryBlock:
        return WorkingMemoryBlock(
            user_id=row["user_id"],
            block_name=row["block_name"],
            content=row["content"],
            updated_at=_coerce_timestamp(row["updated_at"]),
        )

    def _rank_memories(
        self,
        memories: list[Memory],
        query_embedding: list[float],
        limit: int,
    ) -> list[MemorySearchResult]:
        query_vector = _normalize_embedding(query_embedding)
        scored: list[MemorySearchResult] = []
        for memory in memories:
            score = float(np.dot(query_vector, _normalize_embedding(memory.embedding)))
            scored.append(MemorySearchResult(**memory.model_dump(), score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]
