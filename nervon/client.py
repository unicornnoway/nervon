from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from nervon.models import Episode, Memory, MemorySearchResult, WorkingMemoryBlock
from nervon.pipeline import (
    compare_and_decide,
    extract_facts,
    get_embedding,
    summarize_conversation,
)
from nervon.retrieval import ContextAssembler, MemorySearcher
from nervon.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


class MemoryClient:
    def __init__(
        self,
        user_id: str,
        db_path: str = "nervon.db",
        llm_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "gemini/gemini-embedding-001",
        embedding_dim: int = 3072,
    ) -> None:
        self.user_id = user_id
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.storage = SQLiteStorage(db_path)
        self.searcher = MemorySearcher(self.storage, embedding_model)
        self.context_assembler = ContextAssembler(self.storage, self.searcher)

    def add(self, messages: list[dict[str, Any]] | str) -> list[str]:
        normalized_messages = self._normalize_messages(messages)
        facts = extract_facts(normalized_messages, self.llm_model)
        stored_memory_ids: list[str] = []

        for fact in facts:
            try:
                fact_embedding = self._embed_text(fact)
                if not fact_embedding:
                    continue

                existing_memories = self.storage.search_memories(
                    self.user_id,
                    fact_embedding,
                    limit=5,
                )
                decision = compare_and_decide(fact, existing_memories, self.llm_model)
                stored_id = self._apply_decision(decision, fact, fact_embedding)
                if stored_id:
                    stored_memory_ids.append(stored_id)
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.warning("Failed to process fact %r: %s", fact, exc)

        self._store_episode(normalized_messages)
        return stored_memory_ids

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        return self.searcher.search(self.user_id, query, limit=limit)

    def get_context(self, query: str | None = None, max_tokens: int = 2000) -> str:
        return self.context_assembler.get_context(
            self.user_id,
            query=query,
            max_tokens=max_tokens,
        )

    def get_working_memory(self) -> list[WorkingMemoryBlock]:
        return self.storage.get_working_memory(self.user_id)

    def set_working_memory(self, block_name: str, content: str) -> None:
        self.storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id=self.user_id,
                block_name=block_name,
                content=content,
            )
        )

    def get_episodes(self, limit: int = 10) -> list[Episode]:
        return self.storage.get_episodes(self.user_id)[:limit]

    def reset(self) -> None:
        self.storage.delete_user_data(self.user_id)

    def close(self) -> None:
        self.storage.close()

    def _apply_decision(
        self,
        decision: dict[str, str | None],
        fact: str,
        fact_embedding: list[float],
    ) -> str | None:
        action = decision.get("action")
        memory_id = decision.get("memory_id")
        content = decision.get("content")
        resolved_content = content if isinstance(content, str) and content.strip() else fact
        now = datetime.now(timezone.utc)

        if action == "ADD":
            memory = Memory(
                user_id=self.user_id,
                content=resolved_content,
                embedding=fact_embedding if resolved_content == fact else self._embed_text(resolved_content),
                valid_from=now,
                created_at=now,
                embedding_model=self.embedding_model,
            )
            self.storage.add_memory(memory)
            return memory.id

        if action == "UPDATE" and isinstance(memory_id, str):
            replacement_embedding = (
                fact_embedding
                if resolved_content == fact
                else self._embed_text(resolved_content)
            )
            if not replacement_embedding:
                return None
            self.storage.retire_memory(memory_id, now)
            memory = Memory(
                user_id=self.user_id,
                content=resolved_content,
                embedding=replacement_embedding,
                valid_from=now,
                created_at=now,
                embedding_model=self.embedding_model,
            )
            self.storage.add_memory(memory)
            return memory.id

        if action == "DELETE" and isinstance(memory_id, str):
            self.storage.retire_memory(memory_id, now)
            return None

        return None

    def _store_episode(self, messages: list[dict[str, Any]]) -> None:
        episode_payload = summarize_conversation(messages, self.llm_model)
        summary = episode_payload.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            return

        summary_embedding = self._embed_text(summary)
        if not summary_embedding:
            return

        key_topics = episode_payload.get("key_topics", [])
        episode = Episode(
            user_id=self.user_id,
            summary=summary.strip(),
            key_topics=key_topics if isinstance(key_topics, list) else [],
            embedding=summary_embedding,
            message_count=len(messages),
        )
        self.storage.add_episode(episode)

    def _embed_text(self, text: str, task_type: str | None = "RETRIEVAL_DOCUMENT") -> list[float]:
        embedding = get_embedding(text, self.embedding_model, task_type=task_type)
        if not embedding:
            return []
        if self.embedding_dim and len(embedding) != self.embedding_dim:
            logger.warning(
                "Embedding dimension mismatch for model %s: expected %s got %s",
                self.embedding_model,
                self.embedding_dim,
                len(embedding),
            )
            return []
        return embedding

    def _embed_query(self, text: str) -> list[float]:
        """Embed a search query with RETRIEVAL_QUERY task type."""
        return self._embed_text(text, task_type="RETRIEVAL_QUERY")

    def _normalize_messages(
        self, messages: list[dict[str, Any]] | str
    ) -> list[dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            normalized.append({"role": role, "content": content})
        return normalized
