from __future__ import annotations

from neurai.models import Memory
from neurai.pipeline._utils import format_messages

FACT_EXTRACTION_PROMPT = """You extract durable, atomic user facts from a conversation.

Rules:
- Return only facts that are explicitly stated or strongly implied by the user.
- Prefer standalone facts that can be stored independently.
- Keep facts concise and in natural language.
- Preserve temporal changes when relevant, such as moves, job changes, or relationship updates.
- Do not include assistant suggestions, plans, or speculative statements.
- Do not duplicate equivalent facts.
- Return valid JSON only in the shape {"facts": ["fact1", "fact2"]}.

Example 1:
Conversation:
1. user: I live in San Francisco now.
2. assistant: Nice. Do you still work at Stripe?
3. user: No, I joined Figma in January.
Output:
{"facts": ["User lives in San Francisco.", "User joined Figma in January.", "User no longer works at Stripe."]}

Example 2:
Conversation:
1. user: My favorite coffee is pour-over.
2. assistant: Noted. Anything else?
3. user: I usually wake up around 6am.
Output:
{"facts": ["User's favorite coffee is pour-over.", "User usually wakes up around 6am."]}
"""

MEMORY_COMPARISON_PROMPT = """You decide how a new fact relates to existing memories.

Actions:
- ADD: new fact is novel and should be stored as a new memory.
- UPDATE: new fact changes or supersedes an existing memory. Set id to the memory's integer temp ID and set content to the best merged replacement text.
- DELETE: new fact says an existing memory is incorrect or no longer true and the old memory should be retired without a replacement. Set id to the affected integer temp ID.
- NOOP: the new fact is already covered by an existing memory or should not be stored.

Rules:
- Only use IDs provided in the memory list.
- Never invent IDs or UUIDs.
- Prefer UPDATE when a memory changes state and a replacement fact should exist.
- Prefer DELETE only when the old memory should be retired and no replacement memory should be written from this fact.
- Prefer NOOP when the information is already known with no meaningful improvement.
- Return valid JSON only in the shape {"action": "ADD|UPDATE|DELETE|NOOP", "id": int|null, "content": "text"}.

Example 1:
New fact: User lives in San Francisco.
Existing memories:
1. User lives in New York.
Output:
{"action": "UPDATE", "id": 1, "content": "User lives in San Francisco."}

Example 2:
New fact: User enjoys trail running.
Existing memories:
1. User lives in San Francisco.
2. User works at Figma.
Output:
{"action": "ADD", "id": null, "content": "User enjoys trail running."}

Example 3:
New fact: User still works at Figma.
Existing memories:
1. User works at Figma.
Output:
{"action": "NOOP", "id": 1, "content": ""}
"""

EPISODE_SUMMARY_PROMPT = """You summarize a conversation for episodic memory storage.

Rules:
- Write a brief summary focused on what happened or what was learned.
- Keep key_topics short, specific, and useful for retrieval.
- Exclude filler and meta chat.
- Return valid JSON only in the shape {"summary": "...", "key_topics": ["topic1", "topic2"]}.

Example:
Conversation:
1. user: I'm moving from New York to San Francisco next month.
2. assistant: That's a big move. Are you changing jobs too?
3. user: Yes, I'm joining Figma.
Output:
{"summary": "The user discussed an upcoming move from New York to San Francisco and a job change to Figma.", "key_topics": ["move", "San Francisco", "Figma", "job change"]}
"""


def build_fact_extraction_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FACT_EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{format_messages(messages)}",
        },
    ]


def build_memory_comparison_messages(
    fact: str, existing_memories: list[tuple[int, Memory]]
) -> list[dict[str, str]]:
    memory_lines = [f"{temp_id}. {memory.content}" for temp_id, memory in existing_memories]
    memory_text = "\n".join(memory_lines) if memory_lines else "(none)"
    return [
        {"role": "system", "content": MEMORY_COMPARISON_PROMPT},
        {
            "role": "user",
            "content": f"New fact: {fact}\nExisting memories:\n{memory_text}",
        },
    ]


def build_episode_summary_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EPISODE_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{format_messages(messages)}",
        },
    ]
