from __future__ import annotations

from nervon.client import MemoryClient


def _embed_text(text: str, _model: str, task_type: str | None = None) -> list[float]:
    normalized = text.lower()
    if "new york" in normalized or "san francisco" in normalized or "where does user live" in normalized:
        return [1.0, 0.0, 0.0]
    if "python" in normalized or "programming" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _extract_facts(messages: list[dict], _llm_model: str, reference_time: str | None = None) -> list[str]:
    conversation = " ".join(str(message.get("content", "")) for message in messages)
    if "I live in New York and I love Python" in conversation:
        return ["User lives in New York.", "User loves Python."]
    if "I moved to San Francisco" in conversation:
        return ["User lives in San Francisco."]
    return []


def _compare_and_decide(
    fact: str,
    existing_memories: list,
    _llm_model: str,
) -> dict[str, str | None]:
    if "San Francisco" in fact:
        for memory in existing_memories:
            if "New York" in memory.content:
                return {
                    "action": "UPDATE",
                    "memory_id": memory.id,
                    "content": "User lives in San Francisco.",
                }
    return {"action": "ADD", "memory_id": None, "content": fact}


def _summarize_conversation(messages: list[dict], _llm_model: str) -> dict[str, str | list[str]]:
    conversation = " ".join(str(message.get("content", "")) for message in messages)
    if "I live in New York and I love Python" in conversation:
        return {
            "summary": "The user said they live in New York and love Python.",
            "key_topics": ["New York", "Python"],
        }
    if "I moved to San Francisco" in conversation:
        return {
            "summary": "The user moved to San Francisco.",
            "key_topics": ["San Francisco", "move"],
        }
    return {"summary": "", "key_topics": []}


def test_full_conversation_flow_with_mocked_llm(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("nervon.client.extract_facts", _extract_facts)
    monkeypatch.setattr("nervon.client.compare_and_decide", _compare_and_decide)
    monkeypatch.setattr("nervon.client.summarize_conversation", _summarize_conversation)
    monkeypatch.setattr("nervon.client.get_embedding", _embed_text)
    monkeypatch.setattr("nervon.retrieval.search.get_embedding", _embed_text)

    client = MemoryClient("u1", db_path=str(tmp_path / "nervon.db"), embedding_dim=3)
    try:
        initial_ids = client.add("I live in New York and I love Python")

        assert len(initial_ids) == 2
        initial_memories = client.storage.get_memories("u1")
        assert {memory.content for memory in initial_memories} == {
            "User lives in New York.",
            "User loves Python.",
        }

        location_results = client.search("where does user live")
        assert location_results
        assert location_results[0].content == "User lives in New York."

        programming_context = client.get_context("programming")
        assert "User loves Python." in programming_context
        assert "The user said they live in New York and love Python." in programming_context

        updated_ids = client.add("I moved to San Francisco")

        assert len(updated_ids) == 1
        updated_results = client.search("where does user live")
        assert updated_results
        assert updated_results[0].content == "User lives in San Francisco."

        all_memories = client.storage.get_memories("u1", include_retired=True)
        retired_memory = next(
            memory for memory in all_memories if memory.content == "User lives in New York."
        )
        assert retired_memory.valid_until is not None
        assert any(
            memory.content == "User lives in San Francisco." and memory.valid_until is None
            for memory in all_memories
        )

        updated_context = client.get_context("where does user live")
        assert "User lives in San Francisco." in updated_context
        assert "User lives in New York." not in updated_context

        client.reset()

        assert client.search("where does user live") == []
        assert client.get_working_memory() == []
        assert client.get_episodes() == []
        assert client.storage.get_memories("u1", include_retired=True) == []
    finally:
        client.close()
