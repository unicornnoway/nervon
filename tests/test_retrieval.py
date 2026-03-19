from __future__ import annotations

from datetime import datetime, timedelta, timezone

from neurai.models import Episode, Memory, WorkingMemoryBlock
from neurai.retrieval.context import ContextAssembler
from neurai.retrieval.search import MemorySearcher
from neurai.storage.sqlite import SQLiteStorage


def test_search_returns_ranked_results(tmp_path, monkeypatch) -> None:
    storage = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        storage.add_memory(
            Memory(
                user_id="u1",
                content="User lives in San Francisco.",
                embedding=[1.0, 0.0],
                embedding_model="text-embedding-3-small",
            )
        )
        storage.add_memory(
            Memory(
                user_id="u1",
                content="User likes cooking.",
                embedding=[0.0, 1.0],
                embedding_model="text-embedding-3-small",
            )
        )
        monkeypatch.setattr(
            "neurai.retrieval.search.get_embedding",
            lambda text, model: [1.0, 0.0],
        )

        searcher = MemorySearcher(storage, "text-embedding-3-small")
        results = searcher.search("u1", "Where does the user live?", limit=2)

        assert len(results) == 2
        assert results[0].content == "User lives in San Francisco."
        assert results[0].score > results[1].score
    finally:
        storage.close()


def test_get_context_assembles_all_sections(tmp_path, monkeypatch) -> None:
    storage = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id="u1",
                block_name="profile",
                content="Name: Russ",
            )
        )
        storage.add_memory(
            Memory(
                user_id="u1",
                content="User lives in San Francisco.",
                embedding=[1.0, 0.0],
                embedding_model="text-embedding-3-small",
            )
        )
        storage.add_episode(
            Episode(
                user_id="u1",
                summary="The user discussed moving to San Francisco.",
                key_topics=["move", "San Francisco"],
                embedding=[1.0, 0.0],
                occurred_at=datetime.now(timezone.utc) - timedelta(days=1),
                message_count=4,
            )
        )
        monkeypatch.setattr(
            "neurai.retrieval.search.get_embedding",
            lambda text, model: [1.0, 0.0],
        )

        searcher = MemorySearcher(storage, "text-embedding-3-small")
        assembler = ContextAssembler(storage, searcher)
        context = assembler.get_context("u1", query="Where does the user live?")

        assert "WORKING MEMORY" in context
        assert "RELEVANT MEMORIES" in context
        assert "RECENT CONTEXT" in context
        assert "- profile: Name: Russ" in context
        assert "User lives in San Francisco." in context
        assert "The user discussed moving to San Francisco." in context
    finally:
        storage.close()


def test_get_context_with_no_data_returns_empty_string(tmp_path, monkeypatch) -> None:
    storage = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        monkeypatch.setattr(
            "neurai.retrieval.search.get_embedding",
            lambda text, model: [1.0, 0.0],
        )
        searcher = MemorySearcher(storage, "text-embedding-3-small")
        assembler = ContextAssembler(storage, searcher)

        assert assembler.get_context("missing-user", query="anything") == ""
    finally:
        storage.close()


def test_get_context_with_only_working_memory(tmp_path, monkeypatch) -> None:
    storage = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id="u1",
                block_name="preferences",
                content="Coffee: pour-over",
            )
        )
        monkeypatch.setattr(
            "neurai.retrieval.search.get_embedding",
            lambda text, model: [1.0, 0.0],
        )
        searcher = MemorySearcher(storage, "text-embedding-3-small")
        assembler = ContextAssembler(storage, searcher)
        context = assembler.get_context("u1")

        assert "WORKING MEMORY" in context
        assert "RELEVANT MEMORIES" not in context
        assert "RECENT CONTEXT" not in context
        assert "- preferences: Coffee: pour-over" in context
    finally:
        storage.close()


def test_max_tokens_truncates_memories_before_episodes(tmp_path, monkeypatch) -> None:
    storage = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id="u1",
                block_name="profile",
                content="Name: Russ",
            )
        )
        for index in range(3):
            storage.add_memory(
                Memory(
                    user_id="u1",
                    content=f"Memory item {index} with additional detail for truncation.",
                    embedding=[1.0 - (index * 0.1), index * 0.1 + 0.01],
                    embedding_model="text-embedding-3-small",
                )
            )
        now = datetime.now(timezone.utc)
        for index in range(3):
            storage.add_episode(
                Episode(
                    user_id="u1",
                    summary=f"Episode {index} summary with enough text to make truncation visible.",
                    key_topics=[f"topic-{index}"],
                    embedding=[1.0, 0.0],
                    occurred_at=now - timedelta(days=index),
                    message_count=3,
                )
            )
        monkeypatch.setattr(
            "neurai.retrieval.search.get_embedding",
            lambda text, model: [1.0, 0.0],
        )

        searcher = MemorySearcher(storage, "text-embedding-3-small")
        assembler = ContextAssembler(storage, searcher)
        full_context = assembler.get_context("u1", query="query", max_tokens=10_000)
        truncated_context = assembler.get_context("u1", query="query", max_tokens=70)

        assert "RECENT CONTEXT" in truncated_context
        assert "WORKING MEMORY" in truncated_context
        assert full_context.count("Memory item") > truncated_context.count("Memory item")
    finally:
        storage.close()
