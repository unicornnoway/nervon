from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from neurai.models import Episode, Memory, MemorySearchResult, WorkingMemoryBlock


def test_memory_defaults_and_validation() -> None:
    memory = Memory(
        user_id="user-1",
        content="Lives in San Francisco",
        embedding=[0.1, 0.2, 0.3],
        embedding_model="text-embedding-3-small",
    )

    assert memory.id
    assert memory.created_at.tzinfo is not None
    assert memory.valid_from.tzinfo is not None
    assert memory.valid_until is None
    assert memory.embedding == [0.1, 0.2, 0.3]


def test_memory_rejects_invalid_temporal_range() -> None:
    valid_from = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        Memory(
            user_id="user-1",
            content="Lives in SF",
            embedding=[1.0, 0.0],
            embedding_model="text-embedding-3-small",
            valid_from=valid_from,
            valid_until=valid_from - timedelta(seconds=1),
        )


def test_episode_normalizes_topics() -> None:
    episode = Episode(
        user_id="user-1",
        summary="Discussed relocation plans.",
        key_topics=[" relocation ", "", "career"],
        embedding=[0.2, 0.8],
        message_count=4,
    )

    assert episode.key_topics == ["relocation", "career"]


def test_working_memory_requires_content() -> None:
    with pytest.raises(ValidationError):
        WorkingMemoryBlock(user_id="user-1", block_name="profile", content="")


def test_memory_search_result_extends_memory() -> None:
    result = MemorySearchResult(
        user_id="user-1",
        content="Works remotely",
        embedding=[0.4, 0.6],
        embedding_model="text-embedding-3-small",
        score=0.95,
    )

    assert result.score == pytest.approx(0.95)
    assert result.content == "Works remotely"
