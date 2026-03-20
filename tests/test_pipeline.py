from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from nervon.models import Memory
from nervon.pipeline.compare import compare_and_decide
from nervon.pipeline.embeddings import get_embedding, get_embeddings
from nervon.pipeline.extract import extract_facts
from nervon.pipeline.schemas import FactExtractionResponse, MemoryComparisonResponse, EpisodeSummaryResponse
from nervon.pipeline.prompts import (
    FACT_EXTRACTION_PROMPT_TEMPLATE,
    MEMORY_COMPARISON_PROMPT,
    EPISODE_SUMMARY_PROMPT_TEMPLATE,
    build_episode_summary_messages,
    build_fact_extraction_messages,
    build_memory_comparison_messages,
)
from nervon.pipeline.summarize import summarize_conversation


def make_completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors]
    )


def test_prompt_builders_include_expected_content() -> None:
    messages = [
        {"role": "user", "content": "I moved to San Francisco."},
        {"role": "assistant", "content": "Noted."},
    ]
    memories = [
        (
            1,
            Memory(
                user_id="u1",
                content="User lives in New York.",
                embedding=[1.0, 0.0],
                embedding_model="text-embedding-3-small",
            ),
        )
    ]

    fact_prompt = build_fact_extraction_messages(messages)
    compare_prompt = build_memory_comparison_messages(
        "User lives in San Francisco.", memories
    )
    summary_prompt = build_episode_summary_messages(messages)

    assert "REFERENCE_TIME" in fact_prompt[0]["content"]
    assert "atomic facts about ALL participants" in fact_prompt[0]["content"]
    assert "1. user: I moved to San Francisco." in fact_prompt[1]["content"]
    assert MEMORY_COMPARISON_PROMPT in compare_prompt[0]["content"]
    assert "User lives in New York." in compare_prompt[1]["content"]
    assert "REFERENCE_TIME" in summary_prompt[0]["content"]
    assert "2. assistant: Noted." in summary_prompt[1]["content"]


@patch("nervon.pipeline.extract.litellm.completion")
def test_extract_facts_returns_fact_list(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"facts": ["User lives in San Francisco.", "User works at Figma."]}'
    )

    facts = extract_facts(
        [{"role": "user", "content": "I live in San Francisco and work at Figma."}],
        llm_model="openai/gpt-4o-mini",
    )

    assert facts == ["User lives in San Francisco.", "User works at Figma."]
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["response_format"] == FactExtractionResponse


@patch("nervon.pipeline.extract.litellm.completion")
def test_extract_facts_recovers_json_embedded_in_text(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        'Result:\n```json\n{"facts": ["User likes hiking."]}\n```'
    )

    facts = extract_facts(
        [{"role": "user", "content": "I like hiking."}],
        llm_model="openai/gpt-4o-mini",
    )

    assert facts == ["User likes hiking."]


@patch("nervon.pipeline.extract.litellm.completion")
def test_extract_facts_handles_bad_json(mock_completion) -> None:
    mock_completion.return_value = make_completion_response("{not valid json")

    facts = extract_facts(
        [{"role": "user", "content": "I like hiking."}],
        llm_model="openai/gpt-4o-mini",
    )

    assert facts == []


def test_compare_and_decide_without_existing_memories_defaults_to_add() -> None:
    decision = compare_and_decide(
        "User likes hiking.", [], llm_model="openai/gpt-4o-mini"
    )

    assert decision == {"action": "ADD", "memory_id": None, "content": "User likes hiking."}


@patch("nervon.pipeline.compare.litellm.completion")
def test_compare_and_decide_add(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"action": "ADD", "id": null, "content": "User likes hiking."}'
    )

    existing = [
        Memory(
            user_id="u1",
            content="User lives in San Francisco.",
            embedding=[1.0, 0.0],
            embedding_model="text-embedding-3-small",
        )
    ]

    decision = compare_and_decide(
        "User likes hiking.", existing, llm_model="openai/gpt-4o-mini"
    )

    assert decision == {"action": "ADD", "memory_id": None, "content": "User likes hiking."}


@patch("nervon.pipeline.compare.litellm.completion")
def test_compare_and_decide_update_maps_temp_id_to_uuid(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"action": "UPDATE", "id": 1, "content": "User lives in San Francisco."}'
    )

    memory = Memory(
        user_id="u1",
        content="User lives in New York.",
        embedding=[1.0, 0.0],
        embedding_model="text-embedding-3-small",
    )

    decision = compare_and_decide(
        "User moved to San Francisco.", [memory], llm_model="openai/gpt-4o-mini"
    )

    assert decision == {
        "action": "UPDATE",
        "memory_id": memory.id,
        "content": "User lives in San Francisco.",
    }


@patch("nervon.pipeline.compare.litellm.completion")
def test_compare_and_decide_delete(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"action": "DELETE", "id": 1, "content": ""}'
    )

    memory = Memory(
        user_id="u1",
        content="User owns a car.",
        embedding=[1.0, 0.0],
        embedding_model="text-embedding-3-small",
    )

    decision = compare_and_decide(
        "User no longer owns a car.", [memory], llm_model="openai/gpt-4o-mini"
    )

    assert decision == {"action": "DELETE", "memory_id": memory.id, "content": ""}


@patch("nervon.pipeline.compare.litellm.completion")
def test_compare_and_decide_noop(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"action": "NOOP", "id": 1, "content": ""}'
    )

    memory = Memory(
        user_id="u1",
        content="User works at Figma.",
        embedding=[1.0, 0.0],
        embedding_model="text-embedding-3-small",
    )

    decision = compare_and_decide(
        "User works at Figma.", [memory], llm_model="openai/gpt-4o-mini"
    )

    assert decision == {"action": "NOOP", "memory_id": memory.id, "content": ""}


@patch("nervon.pipeline.compare.litellm.completion")
def test_compare_and_decide_handles_bad_json(mock_completion) -> None:
    mock_completion.return_value = make_completion_response("not-json")

    memory = Memory(
        user_id="u1",
        content="User works at Figma.",
        embedding=[1.0, 0.0],
        embedding_model="text-embedding-3-small",
    )

    decision = compare_and_decide(
        "User works at Figma.", [memory], llm_model="openai/gpt-4o-mini"
    )

    assert decision == {"action": "NOOP", "memory_id": None, "content": ""}


@patch("nervon.pipeline.summarize.litellm.completion")
def test_summarize_conversation_returns_summary_and_topics(mock_completion) -> None:
    mock_completion.return_value = make_completion_response(
        '{"summary": "The user discussed a move to San Francisco.", "key_topics": ["move", "San Francisco"]}'
    )

    result = summarize_conversation(
        [{"role": "user", "content": "I am moving to San Francisco."}],
        llm_model="openai/gpt-4o-mini",
    )

    assert result == {
        "summary": "The user discussed a move to San Francisco.",
        "key_topics": ["move", "San Francisco"],
    }


@patch("nervon.pipeline.summarize.litellm.completion")
def test_summarize_conversation_handles_bad_json(mock_completion) -> None:
    mock_completion.return_value = make_completion_response("{oops")

    result = summarize_conversation(
        [{"role": "user", "content": "I am moving to San Francisco."}],
        llm_model="openai/gpt-4o-mini",
    )

    assert result == {"summary": "", "key_topics": []}


@patch("nervon.pipeline.embeddings.litellm.embedding")
def test_get_embedding_returns_vector(mock_embedding) -> None:
    mock_embedding.return_value = make_embedding_response([[0.1, 0.2, 0.3]])

    result = get_embedding("hello", "text-embedding-3-small")

    assert result == [0.1, 0.2, 0.3]
    kwargs = mock_embedding.call_args.kwargs
    assert kwargs == {"model": "text-embedding-3-small", "input": ["hello"]}


@patch("nervon.pipeline.embeddings.litellm.embedding")
def test_get_embeddings_returns_vectors(mock_embedding) -> None:
    mock_embedding.return_value = make_embedding_response([[0.1, 0.2], [0.3, 0.4]])

    result = get_embeddings(["a", "b"], "text-embedding-3-small")

    assert result == [[0.1, 0.2], [0.3, 0.4]]


@patch("nervon.pipeline.embeddings.litellm.embedding")
def test_get_embeddings_handles_errors(mock_embedding) -> None:
    mock_embedding.side_effect = RuntimeError("boom")

    assert get_embedding("hello", "text-embedding-3-small") == []
    assert get_embeddings(["a", "b"], "text-embedding-3-small") == []
