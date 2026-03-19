from __future__ import annotations

from neurai.models import Episode, MemorySearchResult
from neurai.pipeline.embeddings import get_embedding
from neurai.storage.base import StorageBackend


class MemorySearcher:
    def __init__(self, storage: StorageBackend, embedding_model: str) -> None:
        self.storage = storage
        self.embedding_model = embedding_model

    def search(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[MemorySearchResult]:
        query_embedding = get_embedding(query, self.embedding_model)
        if not query_embedding:
            return []
        return self.storage.search_memories(user_id, query_embedding, limit=limit)

    def search_episodes(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[tuple[Episode, float]]:
        query_embedding = get_embedding(query, self.embedding_model)
        if not query_embedding:
            return []
        return self.storage.search_episodes(user_id, query_embedding, limit=limit)
