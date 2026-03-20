from __future__ import annotations

import logging

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class _LiteLLMStub:
        @staticmethod
        def completion(*args, **kwargs):
            raise ModuleNotFoundError("litellm is required for completion requests")

    litellm = _LiteLLMStub()

from nervon.models import Memory
from nervon.pipeline._utils import extract_json_object, extract_message_content, llm_completion_with_retry
from nervon.pipeline.prompts import build_memory_comparison_messages
from nervon.pipeline.schemas import MemoryComparisonResponse

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"ADD", "UPDATE", "DELETE", "NOOP"}

_NOOP_RESULT: dict[str, str | None] = {"action": "NOOP", "memory_id": None, "content": ""}


def compare_and_decide(
    fact: str, existing_memories: list[Memory], llm_model: str
) -> dict[str, str | None]:
    if not existing_memories:
        return {"action": "ADD", "memory_id": None, "content": fact}

    indexed_memories = list(enumerate(existing_memories, start=1))
    id_mapping = {temp_id: memory.id for temp_id, memory in indexed_memories}

    try:
        response = llm_completion_with_retry(
            model=llm_model,
            messages=build_memory_comparison_messages(fact, indexed_memories),
            response_format=MemoryComparisonResponse,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Memory comparison request failed: %s", exc)
        return dict(_NOOP_RESULT)

    raw_content = extract_message_content(response)

    # Try Pydantic parsing first
    parsed = _parse_response(raw_content)
    if parsed is not None:
        return _build_decision(parsed.action, parsed.id, parsed.content, fact, id_mapping)

    # Fallback: legacy JSON extraction + json-repair
    payload = extract_json_object(raw_content)
    if payload is None:
        logger.warning("Memory comparison returned malformed JSON")
        return dict(_NOOP_RESULT)

    action = str(payload.get("action", "NOOP")).strip().upper()
    if action not in VALID_ACTIONS:
        logger.warning("Memory comparison returned invalid action: %s", action)
        action = "NOOP"

    temp_id = payload.get("id")
    content = payload.get("content", "")
    content = content.strip() if isinstance(content, str) else ""

    return _build_decision(action, temp_id, content, fact, id_mapping)


def _parse_response(content: str) -> MemoryComparisonResponse | None:
    """Attempt to parse content as a MemoryComparisonResponse."""
    try:
        return MemoryComparisonResponse.model_validate_json(content)
    except Exception:
        return None


def _build_decision(
    action: str, temp_id: int | None, content: str, fact: str, id_mapping: dict[int, str | None]
) -> dict[str, str | None]:
    real_memory_id = id_mapping.get(temp_id) if isinstance(temp_id, int) else None
    content = content.strip() if isinstance(content, str) else ""

    if action == "ADD":
        return {"action": "ADD", "memory_id": None, "content": content or fact}
    if action == "UPDATE":
        if real_memory_id is None:
            logger.warning("Memory comparison returned UPDATE without a valid id")
            return dict(_NOOP_RESULT)
        return {
            "action": "UPDATE",
            "memory_id": real_memory_id,
            "content": content or fact,
        }
    if action == "DELETE":
        if real_memory_id is None:
            logger.warning("Memory comparison returned DELETE without a valid id")
            return dict(_NOOP_RESULT)
        return {"action": "DELETE", "memory_id": real_memory_id, "content": ""}
    return {"action": "NOOP", "memory_id": real_memory_id, "content": ""}
