from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from neurai.models import Episode, Memory, WorkingMemoryBlock
from neurai.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    backend = SQLiteStorage(str(tmp_path / "neurai.db"))
    try:
        yield backend
    finally:
        backend.close()


def test_memory_crud_and_temporal_filtering(storage: SQLiteStorage) -> None:
    retired_at = datetime.now(timezone.utc)
    active_memory = Memory(
        user_id="u1",
        content="Lives in San Francisco",
        embedding=[1.0, 0.0, 0.0],
        embedding_model="text-embedding-3-small",
    )
    retired_memory = Memory(
        user_id="u1",
        content="Previously lived in New York",
        embedding=[0.8, 0.2, 0.0],
        embedding_model="text-embedding-3-small",
        valid_from=retired_at - timedelta(days=10),
    )

    storage.add_memory(active_memory)
    storage.add_memory(retired_memory)
    storage.retire_memory(retired_memory.id, retired_at)

    fetched = storage.get_memory(active_memory.id)
    assert fetched is not None
    assert fetched.content == active_memory.content

    current_memories = storage.get_memories("u1")
    all_memories = storage.get_memories("u1", include_retired=True)

    assert [memory.id for memory in current_memories] == [active_memory.id]
    assert {memory.id for memory in all_memories} == {
        active_memory.id,
        retired_memory.id,
    }
    retired = next(memory for memory in all_memories if memory.id == retired_memory.id)
    assert retired.valid_until == retired_at


def test_vector_search_ranks_closest_memory_first(storage: SQLiteStorage) -> None:
    best = Memory(
        user_id="u1",
        content="Enjoys hiking in the mountains",
        embedding=[1.0, 0.0],
        embedding_model="text-embedding-3-small",
    )
    medium = Memory(
        user_id="u1",
        content="Likes outdoor travel",
        embedding=[0.6, 0.4],
        embedding_model="text-embedding-3-small",
    )
    worst = Memory(
        user_id="u1",
        content="Prefers staying indoors",
        embedding=[0.0, 1.0],
        embedding_model="text-embedding-3-small",
    )

    storage.add_memory(best)
    storage.add_memory(medium)
    storage.add_memory(worst)

    results = storage.search_memories("u1", [0.95, 0.05], limit=3)

    assert [result.id for result in results] == [best.id, medium.id, worst.id]
    assert results[0].score > results[1].score > results[2].score


def test_episode_crud_and_search(storage: SQLiteStorage) -> None:
    older = Episode(
        user_id="u1",
        summary="Talked about a backend bug.",
        key_topics=["backend", "bugfix"],
        embedding=[1.0, 0.0],
        occurred_at=datetime.now(timezone.utc) - timedelta(days=2),
        message_count=6,
    )
    newer = Episode(
        user_id="u1",
        summary="Planned the storage layer.",
        key_topics=["storage", "sqlite"],
        embedding=[0.0, 1.0],
        occurred_at=datetime.now(timezone.utc) - timedelta(days=1),
        message_count=8,
    )

    storage.add_episode(older)
    storage.add_episode(newer)

    episodes = storage.get_episodes("u1")
    assert [episode.id for episode in episodes] == [newer.id, older.id]

    filtered = storage.get_episodes(
        "u1",
        after=datetime.now(timezone.utc) - timedelta(days=1, hours=12),
    )
    assert [episode.id for episode in filtered] == [newer.id]

    ranked = storage.search_episodes("u1", [0.0, 1.0], limit=2)
    assert [episode.id for episode, _score in ranked] == [newer.id, older.id]
    assert ranked[0][1] > ranked[1][1]


def test_working_memory_crud_and_cap(storage: SQLiteStorage) -> None:
    for index in range(10):
        storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id="u1",
                block_name=f"block-{index}",
                content=f"content-{index}",
            )
        )

    blocks = storage.get_working_memory("u1")
    assert len(blocks) == 10

    storage.upsert_working_memory(
        WorkingMemoryBlock(
            user_id="u1",
            block_name="block-0",
            content="updated",
        )
    )
    updated = next(block for block in storage.get_working_memory("u1") if block.block_name == "block-0")
    assert updated.content == "updated"

    with pytest.raises(ValueError):
        storage.upsert_working_memory(
            WorkingMemoryBlock(
                user_id="u1",
                block_name="block-10",
                content="overflow",
            )
        )

    storage.delete_working_memory("u1", "block-9")
    remaining_names = {block.block_name for block in storage.get_working_memory("u1")}
    assert "block-9" not in remaining_names
