from __future__ import annotations

from datetime import datetime, timezone

from nervon.models import Memory
from nervon.pipeline._utils import format_messages


def _reference_time() -> str:
    """Return current UTC timestamp in ISO 8601 for prompt injection."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_date() -> str:
    """Return current date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


FACT_EXTRACTION_PROMPT_TEMPLATE = """You extract durable, atomic facts about ALL participants from a conversation.

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Rules:
- Return facts about EVERY person mentioned — not just one speaker.
- Each fact MUST be self-contained — understandable without the original conversation.
- Always include WHO the fact is about (use their name).
- Preserve specific details: full names, dates, numbers, locations, organizations.
- CRITICAL: Convert ALL relative time references to absolute dates using REFERENCE_TIME above.
  - "yesterday" → the actual date (e.g., "2026-03-18")
  - "last week" → the actual date range
  - "next Friday" → the actual date
  - "in January" → "in January 2026" (include year)
  - "3 months ago" → the actual month/year
- If a date/time is mentioned, always include it in the fact.
- Do not include assistant suggestions, plans, or speculative statements.
- Do not duplicate equivalent facts.
- Return ONLY valid JSON in this exact shape: {{"facts": ["fact1", "fact2"]}}

Example 1 (reference time: 2026-03-19):
Conversation:
1. user: I live in San Francisco now.
2. assistant: Nice. Do you still work at Stripe?
3. user: No, I joined Figma in January.
Output:
{{"facts": ["User lives in San Francisco as of March 2026.", "User joined Figma in January 2026.", "User previously worked at Stripe but left before January 2026."]}}

Example 2 (reference time: 2026-03-19):
Conversation:
1. user: I had dinner with Sarah Chen yesterday.
2. assistant: Nice! What did you have?
3. user: We went to Nobu in Tribeca. She mentioned she's moving to London next month.
Output:
{{"facts": ["User had dinner with Sarah Chen on 2026-03-18 at Nobu in Tribeca, NYC.", "Sarah Chen is planning to move to London in April 2026."]}}

Example 3 (reference time: 2026-03-19):
Conversation:
1. user: My son just turned 5.
2. assistant: Happy birthday to him!
3. user: Thanks, we had a party at home. My wife Maria made a dinosaur cake.
Output:
{{"facts": ["User's son turned 5 years old around March 2026.", "User is married to Maria.", "User's son's birthday party was held at home with a dinosaur cake."]}}
"""

MEMORY_COMPARISON_PROMPT = """Decide how a new fact relates to existing memories. Respond with ONE JSON object.

Actions: ADD (novel fact), UPDATE (supersedes existing — set id + new content), DELETE (old memory is wrong/obsolete — set id), NOOP (already known).

Rules:
- Use ONLY IDs from the memory list below.
- UPDATE: merge old + new into the best replacement text. Preserve dates and specifics.
- DELETE: only when old memory should be retired with NO replacement.
- NOOP: information already known, no meaningful change.
- Return ONLY: {"action": "ADD|UPDATE|DELETE|NOOP", "id": INT_OR_NULL, "content": "TEXT_OR_EMPTY"}

{"action": "UPDATE", "id": 1, "content": "User lives in San Francisco as of March 2026."}
{"action": "ADD", "id": null, "content": "User enjoys trail running."}
{"action": "NOOP", "id": null, "content": ""}
{"action": "DELETE", "id": 2, "content": ""}
"""

EPISODE_SUMMARY_PROMPT_TEMPLATE = """You summarize a conversation for episodic memory storage.

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Rules:
- Write a brief summary focused on what happened or what was learned.
- Include specific names, dates, and locations mentioned.
- Convert ALL relative dates to absolute dates using REFERENCE_TIME.
- Keep key_topics short, specific, and useful for retrieval.
- Exclude filler and meta chat.
- Return ONLY valid JSON: {{"summary": "...", "key_topics": ["topic1", "topic2"]}}

Example (reference time: 2026-03-19):
Conversation:
1. user: I'm moving from New York to San Francisco next month.
2. assistant: That's a big move. Are you changing jobs too?
3. user: Yes, I'm joining Figma.
Output:
{{"summary": "User discussed moving from New York to San Francisco in April 2026 and starting a new job at Figma.", "key_topics": ["relocation", "San Francisco", "Figma", "job change", "April 2026"]}}
"""


def build_fact_extraction_messages(messages: list[dict], reference_time: str | None = None) -> list[dict[str, str]]:
    prompt = FACT_EXTRACTION_PROMPT_TEMPLATE.format(reference_time=reference_time or _reference_time())
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Conversation:\n{format_messages(messages)}",
        },
    ]


def build_memory_comparison_messages(
    fact: str, existing_memories: list[tuple[int, Memory]]
) -> list[dict[str, str]]:
    memory_lines = []
    for temp_id, memory in existing_memories:
        ts = memory.valid_from.strftime("%Y-%m-%d") if memory.valid_from else "unknown"
        memory_lines.append(f"{temp_id}. [{ts}] {memory.content}")
    memory_text = "\n".join(memory_lines) if memory_lines else "(none)"
    return [
        {"role": "system", "content": MEMORY_COMPARISON_PROMPT},
        {
            "role": "user",
            "content": f"New fact: {fact}\nExisting memories:\n{memory_text}",
        },
    ]


def build_episode_summary_messages(messages: list[dict], reference_time: str | None = None) -> list[dict[str, str]]:
    prompt = EPISODE_SUMMARY_PROMPT_TEMPLATE.format(reference_time=reference_time or _reference_time())
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Conversation:\n{format_messages(messages)}",
        },
    ]
