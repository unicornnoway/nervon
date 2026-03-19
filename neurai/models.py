from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid4())


Embedding = Annotated[list[float], Field(min_length=1)]


class NeuraiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Memory(NeuraiBaseModel):
    id: str = Field(default_factory=generate_id)
    user_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding: Embedding
    valid_from: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    embedding_model: str = Field(min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("embedding must not be empty")
        return [float(item) for item in value]

    @field_validator("valid_until")
    @classmethod
    def validate_valid_until(
        cls, value: datetime | None, info
    ) -> datetime | None:
        if value is not None:
            valid_from = info.data.get("valid_from")
            if valid_from is not None and value < valid_from:
                raise ValueError("valid_until must be on or after valid_from")
        return value


class Episode(NeuraiBaseModel):
    id: str = Field(default_factory=generate_id)
    user_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_topics: list[str] = Field(default_factory=list)
    embedding: Embedding
    occurred_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)
    message_count: int = Field(ge=0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("embedding must not be empty")
        return [float(item) for item in value]

    @field_validator("key_topics")
    @classmethod
    def validate_topics(cls, value: list[str]) -> list[str]:
        normalized = [topic.strip() for topic in value if topic.strip()]
        return normalized


class WorkingMemoryBlock(NeuraiBaseModel):
    user_id: str = Field(min_length=1)
    block_name: str = Field(min_length=1)
    content: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=utc_now)


class MemorySearchResult(Memory):
    score: float
