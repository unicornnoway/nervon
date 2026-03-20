from __future__ import annotations

from nervon.models import Episode, MemorySearchResult, WorkingMemoryBlock
from nervon.retrieval.search import MemorySearcher
from nervon.storage.base import StorageBackend


class ContextAssembler:
    def __init__(self, storage: StorageBackend, searcher: MemorySearcher) -> None:
        self.storage = storage
        self.searcher = searcher

    def get_context(
        self, user_id: str, query: str | None = None, max_tokens: int = 2000
    ) -> str:
        blocks = self.storage.get_working_memory(user_id)
        memories = self.searcher.search(user_id, query, limit=30) if query else []
        # Use semantic search for episodes when query is available, fall back to recent
        if query:
            episode_results = self.searcher.search_episodes(user_id, query, limit=10)
            episodes = [ep for ep, _score in episode_results]
        else:
            episodes = self.storage.get_episodes(user_id)[:10]

        if not blocks and not memories and not episodes:
            return ""

        working_lines = self._format_working_memory_lines(blocks)
        memory_lines = self._format_memories_lines(memories)
        episode_lines = self._format_episodes_lines(episodes)

        while self._estimate_tokens(
            self._assemble_context(working_lines, memory_lines, episode_lines)
        ) > max_tokens:
            if memory_lines:
                memory_lines.pop()
                continue
            if episode_lines:
                episode_lines.pop()
                continue
            break

        return self._assemble_context(working_lines, memory_lines, episode_lines)

    def format_memories(self, memories: list[MemorySearchResult]) -> str:
        return self._format_section("RELEVANT MEMORIES", self._format_memories_lines(memories))

    def format_episodes(self, episodes: list[Episode]) -> str:
        return self._format_section("RECENT CONTEXT", self._format_episodes_lines(episodes))

    def format_working_memory(self, blocks: list[WorkingMemoryBlock]) -> str:
        return self._format_section("WORKING MEMORY", self._format_working_memory_lines(blocks))

    def _assemble_context(
        self,
        working_lines: list[str],
        memory_lines: list[str],
        episode_lines: list[str],
    ) -> str:
        sections: list[str] = []
        working = self._format_section("WORKING MEMORY", working_lines)
        memories = self._format_section("RELEVANT MEMORIES", memory_lines)
        episodes = self._format_section("RECENT CONTEXT", episode_lines)
        if working:
            sections.append(working)
        if memories:
            sections.append(memories)
        if episodes:
            sections.append(episodes)
        return "\n\n".join(sections)

    def _format_working_memory_lines(
        self, blocks: list[WorkingMemoryBlock]
    ) -> list[str]:
        return [f"- {block.block_name}: {block.content}" for block in blocks]

    def _format_memories_lines(
        self, memories: list[MemorySearchResult]
    ) -> list[str]:
        lines = []
        for memory in memories:
            ts = memory.created_at.strftime("%Y-%m-%d")
            lines.append(f"- [{ts}] {memory.content} (score: {memory.score:.3f})")
        return lines

    def _format_episodes_lines(self, episodes: list[Episode]) -> list[str]:
        lines: list[str] = []
        for episode in episodes:
            occurred = episode.occurred_at.date().isoformat()
            if episode.key_topics:
                topics = ", ".join(episode.key_topics)
                lines.append(f"- {occurred}: {episode.summary} [topics: {topics}]")
            else:
                lines.append(f"- {occurred}: {episode.summary}")
        return lines

    def _format_section(self, header: str, lines: list[str]) -> str:
        if not lines:
            return ""
        return "\n".join([header, *lines])

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)
