# AGENTS.md — Coding Agent Instructions

## Project: Reasoning-Native Memory Framework

A memory SDK for AI agents where LLM reasoning (not just vector similarity) drives what to remember, update, forget, and resolve conflicts. Three-tier architecture: Working Memory, Semantic Store, Episodic Log.

## Tech Stack
- **Language:** Python 3.11+
- **LLM:** litellm (provider-agnostic)
- **Embeddings:** OpenAI text-embedding-3-small (1536 dim) via litellm
- **Storage:** SQLite + numpy for vector search (v1)
- **Dependencies:** litellm, numpy, pydantic, aiosqlite (async support later)
- **Testing:** pytest

## Architecture Overview

### Three Tiers
1. **Working Memory** — Always loaded into agent context. Key-value text blocks. Max 10 blocks per user. No search needed.
2. **Semantic Store** — Facts extracted from conversations. Vector search + LLM re-ranking. Temporal versioning (valid_from/valid_until). 
3. **Episodic Log** — Append-only conversation summaries. Searched by time + optional vector search.

### Write Pipeline (add/process_conversation)
1. **EXTRACT** — LLM extracts atomic facts from conversation
2. **COMPARE** — For each fact, vector search Semantic Store for conflicts
3. **DECIDE** — LLM judges: ADD new / UPDATE existing / DELETE outdated / NOOP
4. **STORE** — New facts stored with valid_from=now. Updated facts: old gets valid_until=now, new created. Simultaneously, conversation summary → Episodic Log.

### Read Pipeline (search/get_context)
- Working Memory: always returned (no search)
- Semantic Store: vector search, filter valid_until IS NULL (current only), optional LLM re-rank
- Episodic Log: temporal filter + optional vector search
- `get_context()` merges all three into prompt-ready text

## Data Models

### Memory (Semantic Store)
```python
class Memory:
    id: str              # UUID v4
    user_id: str
    content: str         # Natural language fact
    embedding: list[float]  # 1536-dim vector
    valid_from: datetime
    valid_until: datetime | None  # None = current
    created_at: datetime
    embedding_model: str  # e.g. "text-embedding-3-small"
```

### Episode (Episodic Log)
```python
class Episode:
    id: str
    user_id: str
    summary: str
    key_topics: list[str]
    embedding: list[float]
    occurred_at: datetime
    created_at: datetime
    message_count: int
```

### WorkingMemoryBlock
```python
class WorkingMemoryBlock:
    user_id: str
    block_name: str     # e.g. "profile", "preferences"
    content: str
    updated_at: datetime
```

## SDK API

```python
client = MemoryClient(storage="sqlite:///mem.db", llm="openai/gpt-4o-mini", embedding="text-embedding-3-small")

# Semantic Store
client.add("I moved from NYC to SF", user_id="u1")  # Extract + Compare + Store
memories = client.search("Where does user live?", user_id="u1")  # Vector search
all_mems = client.get_all(user_id="u1", include_retired=False)

# Episodic Log
client.process_conversation(messages=[...], user_id="u1")  # Dual write: facts + episode
episodes = client.search_episodes(user_id="u1", after=date, query="optional")

# Working Memory
client.set_working_memory(user_id="u1", block="profile", content="Name: Russ, Age: 20")
blocks = client.get_working_memory(user_id="u1")

# Combined retrieval
context = client.get_context(user_id="u1", query="optional query")  # Returns prompt-ready text
```

## Directory Structure
```
neurai/
├── __init__.py
├── client.py          # MemoryClient — main entry point
├── models.py          # Pydantic data models (Memory, Episode, WorkingMemoryBlock)
├── storage/
│   ├── __init__.py
│   ├── base.py        # StorageBackend Protocol
│   ├── sqlite.py      # SQLite implementation (tables + numpy vector search)
│   └── migrations.py  # Schema creation/migration
├── pipeline/
│   ├── __init__.py
│   ├── extract.py     # LLM fact extraction
│   ├── compare.py     # Vector search + LLM comparison/judgment
│   ├── prompts.py     # All LLM prompt templates
│   └── embeddings.py  # Embedding generation wrapper
├── retrieval/
│   ├── __init__.py
│   ├── search.py      # Vector search + filtering
│   ├── rerank.py      # Optional LLM re-ranking
│   └── context.py     # get_context() — merge all tiers into prompt text
└── utils.py           # Helpers (JSON parsing, UUID generation, etc.)
tests/
├── test_models.py
├── test_storage.py
├── test_pipeline.py
├── test_retrieval.py
├── test_client.py     # Integration tests
└── conftest.py        # Fixtures
pyproject.toml
README.md
```

## Error Handling Rules
- LLM returns bad JSON → try to fix, if can't → skip, log warning
- LLM API timeout → skip, log error, don't store partial data
- Vector search returns no results → treat as ADD (new fact)
- Any exception in write pipeline → catch, log, skip that fact, continue with others
- Never store partial/corrupt data

## Key Design Decisions
- UPDATE = retire old (set valid_until=now) + create new. Never overwrite.
- Use integer temp IDs when showing existing memories to LLM (prevent UUID hallucination). See Mem0's temp_uuid_mapping pattern.
- Working Memory is explicit (set by user/agent), not auto-extracted.
- Episodes are append-only, never modified.
- v1: no forgetting/decay, no multi-agent, no graph layer.
- Vector search: cosine similarity via numpy dot product on normalized vectors. No external vector DB dependency.

## LLM Prompt Style
- Use JSON mode (response_format={"type": "json_object"}) for all structured LLM calls
- Few-shot examples in prompts (see Mem0's prompt patterns)
- Extraction prompt: extract atomic facts with temporal hints
- Comparison prompt: given new fact + existing memories → decide ADD/UPDATE/DELETE/NOOP
- Episode summary prompt: summarize conversation + extract key topics

## Testing Strategy
- Unit tests for each module (storage, pipeline, retrieval)
- Integration test: full add → search → update → search cycle
- Scenario tests: "user changed jobs", "user moved cities", "what did we discuss last week?"
- Mock LLM calls in unit tests, real LLM calls in integration tests (optional, behind flag)
