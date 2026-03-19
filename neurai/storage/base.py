from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from neurai.models import Episode, Memory, MemorySearchResult, WorkingMemoryBlock


@runtime_checkable
class StorageBackend(Protocol):
    def add_memory(self, memory: Memory) -> None:
        """Persist a semantic memory."""

    def get_memory(self, memory_id: str) -> Memory | None:
        """Fetch a memory by id."""

    def get_memories(
        self, user_id: str, include_retired: bool = False
    ) -> list[Memory]:
        """Fetch memories for a user."""

    def retire_memory(self, memory_id: str, valid_until: datetime) -> None:
        """Retire a memory by setting valid_until."""

    def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        include_retired: bool = False,
    ) -> list[MemorySearchResult]:
        """Search memories with cosine similarity."""

    def add_episode(self, episode: Episode) -> None:
        """Persist an episodic summary."""

    def get_episodes(
        self,
        user_id: str,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[Episode]:
        """Fetch episodes for a user, optionally by time range."""

    def search_episodes(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 5,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[tuple[Episode, float]]:
        """Search episodes with cosine similarity."""

    def upsert_working_memory(self, block: WorkingMemoryBlock) -> None:
        """Create or update a working memory block."""

    def get_working_memory(self, user_id: str) -> list[WorkingMemoryBlock]:
        """Fetch all working memory blocks for a user."""

    def delete_working_memory(self, user_id: str, block_name: str) -> None:
        """Delete a working memory block."""

    def delete_user_data(self, user_id: str) -> None:
        """Delete all memories, episodes, and working memory for a user."""

    def close(self) -> None:
        """Close any backend resources."""
