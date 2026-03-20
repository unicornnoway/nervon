from __future__ import annotations

from unittest.mock import patch

from nervon.client import MemoryClient
from nervon.models import Episode, Memory


def test_add_with_string_input(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    try:
        with (
            patch("nervon.client.extract_facts", return_value=["User lives in New York."]) as mock_extract,
            patch(
                "nervon.client.get_embedding",
                side_effect=[[1.0, 0.0], [0.0, 1.0]],
            ),
            patch(
                "nervon.client.compare_and_decide",
                return_value={
                    "action": "ADD",
                    "memory_id": None,
                    "content": "User lives in New York.",
                },
            ),
            patch(
                "nervon.client.summarize_conversation",
                return_value={
                    "summary": "The user said they live in New York.",
                    "key_topics": ["New York"],
                },
            ),
        ):
            memory_ids = client.add("I live in New York.")

        assert len(memory_ids) == 1
        mock_extract.assert_called_once_with(
            [{"role": "user", "content": "I live in New York."}],
            "gemini/gemini-2.0-flash",
            reference_time=None,
        )
        assert [memory.content for memory in client.storage.get_memories("u1")] == [
            "User lives in New York."
        ]
        assert len(client.get_episodes()) == 1
    finally:
        client.close()


def test_add_with_message_list(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    messages = [
        {"role": "user", "content": "I live in New York."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "I love Python."},
    ]
    try:
        with (
            patch(
                "nervon.client.extract_facts",
                return_value=["User lives in New York.", "User loves Python."],
            ) as mock_extract,
            patch(
                "nervon.client.get_embedding",
                side_effect=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            ),
            patch(
                "nervon.client.compare_and_decide",
                side_effect=[
                    {
                        "action": "ADD",
                        "memory_id": None,
                        "content": "User lives in New York.",
                    },
                    {
                        "action": "ADD",
                        "memory_id": None,
                        "content": "User loves Python.",
                    },
                ],
            ),
            patch(
                "nervon.client.summarize_conversation",
                return_value={
                    "summary": "The user shared where they live and that they love Python.",
                    "key_topics": ["New York", "Python"],
                },
            ),
        ):
            memory_ids = client.add(messages)

        assert len(memory_ids) == 2
        mock_extract.assert_called_once_with(messages, "gemini/gemini-2.0-flash", reference_time=None)
        assert [memory.content for memory in client.storage.get_memories("u1")] == [
            "User loves Python.",
            "User lives in New York.",
        ]
    finally:
        client.close()


def test_add_handles_update_action(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    original = Memory(
        user_id="u1",
        content="User lives in New York.",
        embedding=[1.0, 0.0],
        embedding_model="openai/text-embedding-3-small",
    )
    client.storage.add_memory(original)
    try:
        with (
            patch(
                "nervon.client.extract_facts",
                return_value=["User lives in San Francisco."],
            ),
            patch(
                "nervon.client.get_embedding",
                side_effect=[[1.0, 0.0], [0.0, 1.0]],
            ),
            patch(
                "nervon.client.compare_and_decide",
                return_value={
                    "action": "UPDATE",
                    "memory_id": original.id,
                    "content": "User lives in San Francisco.",
                },
            ),
            patch(
                "nervon.client.summarize_conversation",
                return_value={
                    "summary": "The user moved to San Francisco.",
                    "key_topics": ["move", "San Francisco"],
                },
            ),
        ):
            memory_ids = client.add("I moved to San Francisco.")

        assert len(memory_ids) == 1
        all_memories = client.storage.get_memories("u1", include_retired=True)
        retired = next(memory for memory in all_memories if memory.id == original.id)
        current = next(memory for memory in all_memories if memory.id == memory_ids[0])
        assert retired.valid_until is not None
        assert current.content == "User lives in San Francisco."
        assert current.valid_until is None
    finally:
        client.close()


def test_add_handles_delete_action(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    original = Memory(
        user_id="u1",
        content="User owns a car.",
        embedding=[1.0, 0.0],
        embedding_model="openai/text-embedding-3-small",
    )
    client.storage.add_memory(original)
    try:
        with (
            patch(
                "nervon.client.extract_facts",
                return_value=["User no longer owns a car."],
            ),
            patch(
                "nervon.client.get_embedding",
                side_effect=[[1.0, 0.0], [0.0, 1.0]],
            ),
            patch(
                "nervon.client.compare_and_decide",
                return_value={
                    "action": "DELETE",
                    "memory_id": original.id,
                    "content": "",
                },
            ),
            patch(
                "nervon.client.summarize_conversation",
                return_value={
                    "summary": "The user said they no longer own a car.",
                    "key_topics": ["car"],
                },
            ),
        ):
            memory_ids = client.add("I sold my car.")

        assert memory_ids == []
        retired = client.storage.get_memory(original.id)
        assert retired is not None
        assert retired.valid_until is not None
        assert client.storage.get_memories("u1") == []
    finally:
        client.close()


def test_search_delegates_properly(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    try:
        expected = ["sentinel"]
        with patch.object(client.searcher, "search", return_value=expected) as mock_search:
            result = client.search("where does user live", limit=3)

        assert result == expected
        mock_search.assert_called_once_with("u1", "where does user live", limit=3)
    finally:
        client.close()


def test_get_context_delegates_properly(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    try:
        with patch.object(
            client.context_assembler,
            "get_context",
            return_value="assembled context",
        ) as mock_context:
            result = client.get_context("python", max_tokens=123)

        assert result == "assembled context"
        mock_context.assert_called_once_with("u1", query="python", max_tokens=123)
    finally:
        client.close()


def test_working_memory_set_and_get(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    try:
        client.set_working_memory("profile", "Name: Russ")
        client.set_working_memory("preferences", "Language: Python")

        blocks = client.get_working_memory()

        assert {block.block_name for block in blocks} == {"profile", "preferences"}
        assert {block.content for block in blocks} == {"Name: Russ", "Language: Python"}
    finally:
        client.close()


def test_reset_clears_everything(tmp_path) -> None:
    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=2)
    client.storage.add_memory(
        Memory(
            user_id="u1",
            content="User lives in New York.",
            embedding=[1.0, 0.0],
            embedding_model="openai/text-embedding-3-small",
        )
    )
    client.storage.add_episode(
        Episode(
            user_id="u1",
            summary="The user talked about New York.",
            key_topics=["New York"],
            embedding=[0.0, 1.0],
            message_count=1,
        )
    )
    client.set_working_memory("profile", "Name: Russ")

    try:
        client.reset()

        assert client.storage.get_memories("u1", include_retired=True) == []
        assert client.get_episodes() == []
        assert client.get_working_memory() == []
    finally:
        client.close()
