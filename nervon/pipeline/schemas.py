"""Pydantic models for structured LLM output in the Nervon pipeline.

These models are passed as ``response_format`` to litellm.completion so that
providers with structured-output support (OpenAI, Anthropic via litellm) return
guaranteed-valid JSON matching the schema.  For providers that fall back to
plain ``json_object`` mode the response is parsed with ``Model.model_validate_json``
and ultimately with the legacy ``extract_json_object`` + json-repair path.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class FactExtractionResponse(BaseModel):
    """Response schema for fact extraction."""
    facts: list[str]


class MemoryComparisonResponse(BaseModel):
    """Response schema for memory comparison decisions."""
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    id: Optional[int] = None
    content: str = ""


class EpisodeSummaryResponse(BaseModel):
    """Response schema for episode/conversation summaries."""
    summary: str
    key_topics: list[str]
