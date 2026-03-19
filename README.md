# Neurai

Reasoning-Native Memory Framework for AI Agents.

A memory SDK where LLM reasoning — not just vector similarity — drives what to remember, what to forget, and how to resolve conflicts.

## Architecture

Three-tier memory based on access patterns:

| Tier | Access Pattern | Example |
|------|---------------|---------|
| **Working Memory** | Always loaded | User name, preferences, active project |
| **Semantic Store** | Search by meaning | "Works at Google", "Allergic to shellfish" |
| **Episodic Log** | Search by time | "Mar 19: discussed promotion", "Mar 15: debugged auth" |

## Key Differentiators

- **Temporal versioning** — memories have `valid_from`/`valid_until`, history is preserved
- **LLM-driven decisions** — reasoning at write, retrieve, and update stages
- **Three-tier architecture** — right memory via the right access pattern
- **Active forgetting** (v1.1) — confidence decay keeps retrieval clean

## Quick Start

```python
from neurai import MemoryClient

client = MemoryClient(
    storage="sqlite:///memories.db",
    llm="openai/gpt-4o-mini",
    embedding="text-embedding-3-small"
)

# Store a memory
client.add("I just moved from NYC to SF", user_id="user_123")

# Search
results = client.search("Where does the user live?", user_id="user_123")

# Get prompt-ready context
context = client.get_context(user_id="user_123", query="current location")
```

## Status

🚧 Under active development. v1 in progress.

## License

MIT
