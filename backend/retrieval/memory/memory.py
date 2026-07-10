"""Conversation memory helpers."""

import tiktoken

from retrieval.stores.chat_store import (
    list_session_messages,
    list_session_turns,
    latest_session_assistant_message,
    list_thread_messages,
    list_thread_turns,
    latest_thread_assistant_message,
)
from retrieval.stores.memory_store import (
    get_session_memory,
    get_thread_memory,
    list_session_turn_entities,
    list_turn_entities,
    save_session_memory,
    save_session_turn_entities,
    save_thread_memory,
    save_turn_entities,
)

try:
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - offline fallback for test environments
    class _FallbackEncoding:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, tokens: list[int]) -> str:
            return bytes(tokens).decode("utf-8", errors="ignore")

    _enc = _FallbackEncoding()


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _cap_history_block(full_block: str, max_tokens: int) -> str:
    """Trim history block to max_tokens by dropping the oldest turns first.

    Preserves the header/footer lines and reconstructs a valid block from
    the most-recent turns that fit within max_tokens.  Returns an empty
    string when max_tokens <= 0.
    """
    if not full_block or max_tokens <= 0:
        return ""
    if _token_count(full_block) <= max_tokens:
        return full_block

    # Split on turn lines.  History block format:
    #   --- CONVERSATION HISTORY ---
    #   Q1: ...
    #   A1: ...
    #   ...
    #   --- END HISTORY ---
    lines = full_block.splitlines()
    # Collect (Qn, An) pairs from the end, stopping when we exceed the budget.
    header = "--- CONVERSATION HISTORY ---"
    footer = "--- END HISTORY ---"
    overhead = _token_count(header + "\n" + footer)
    if overhead >= max_tokens:
        return ""

    # Walk turns from most-recent (last) to oldest.
    turn_pairs: list[tuple[str, str]] = []
    i = len(lines) - 1
    while i >= 0:
        line = lines[i]
        if line.startswith("--- END HISTORY") or line.startswith("--- CONVERSATION HISTORY"):
            i -= 1
            continue
        # Expect Qn / An alternating from the bottom
        if line.startswith("A") and i > 0 and lines[i - 1].startswith("Q"):
            turn_pairs.insert(0, (lines[i - 1], line))
            i -= 2
        else:
            i -= 1

    # Greedily include most-recent turns that fit.
    chosen: list[tuple[str, str]] = []
    used = overhead
    for q, a in reversed(turn_pairs):
        pair_tokens = _token_count(q + "\n" + a + "\n")
        if used + pair_tokens > max_tokens:
            break
        chosen.insert(0, (q, a))
        used += pair_tokens

    if not chosen:
        return ""

    rebuilt = [header]
    for q, a in chosen:
        rebuilt.append(q)
        rebuilt.append(a)
    rebuilt.append(footer)
    return "\n".join(rebuilt)


def _limit_history_block_turns(full_block: str, max_turns: int) -> str:
    """Keep only the most recent *max_turns* Q/A pairs from a history block.

    Any rolling summary section is intentionally dropped so prompt history only
    contains concrete recent turns when history injection is enabled.
    """
    if not full_block or max_turns <= 0:
        return ""

    lines = full_block.splitlines()
    turn_pairs: list[tuple[str, str]] = []
    i = len(lines) - 1
    while i >= 0:
        line = lines[i]
        if line.startswith("--- END HISTORY") or line.startswith("--- CONVERSATION HISTORY"):
            i -= 1
            continue
        if line.startswith("--- CONVERSATION SUMMARY") or line.startswith("--- END SUMMARY"):
            i -= 1
            continue
        if line.startswith("A") and i > 0 and lines[i - 1].startswith("Q"):
            turn_pairs.insert(0, (lines[i - 1], line))
            i -= 2
            continue
        i -= 1

    if not turn_pairs:
        return ""

    chosen = turn_pairs[-max_turns:]
    rebuilt = ["--- CONVERSATION HISTORY ---"]
    for q, a in chosen:
        rebuilt.append(q)
        rebuilt.append(a)
    rebuilt.append("--- END HISTORY ---")
    return "\n".join(rebuilt)


def prepare_history_block(full_block: str, *, max_turns: int, max_tokens: int) -> str:
    """Trim history to recent turns first, then enforce the token cap."""
    limited = _limit_history_block_turns(full_block, max_turns=max_turns)
    return _cap_history_block(limited, max_tokens=max_tokens)


class ConversationMemory:
    """Store bounded query/answer turns for prompt continuity.

    Extended in WS7 to also hold per-turn cited-entity sets so the follow-up
    resolver can access recent files, symbols, routes, env_keys, and services
    without a DB round-trip.
    """

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.turns: list[dict[str, str]] = []
        # WS7: parallel list of entity dicts, one per turn.
        self._turn_entities: list[dict] = []
        self._turn_sources: list[list[dict]] = []

    def add(
        self,
        query: str,
        answer: str,
        resolved_query: str | None = None,
        *,
        entities: dict | None = None,
        rendered_sources: list[dict] | None = None,
        primary_intent: str = "",  # accepted but unused in in-process memory
    ) -> None:
        self.turns.append(
            {
                "query": query,
                "answer": answer,
                "resolved_query": resolved_query or query,
            }
        )
        self._turn_entities.append(entities or {})
        self._turn_sources.append(list(rendered_sources or []))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
            self._turn_entities.pop(0)
            self._turn_sources.pop(0)

    def recent_turn_entities(self, max_turns: int = 8) -> list[dict]:
        """Return the last *max_turns* per-turn entity dicts, oldest first.

        Each entry is a dict with keys: entities (dict with files/symbols/etc.).
        """
        return [
            {"entities": e}
            for e in self._turn_entities[-max_turns:]
        ]

    def latest_rendered_sources(self) -> list[dict]:
        if not self._turn_sources:
            return []
        return list(self._turn_sources[-1])

    def latest_query(self) -> str:
        if not self.turns:
            return ""
        return self.turns[-1].get("query", "")

    def latest_resolved_query(self) -> str:
        if not self.turns:
            return ""
        return self.turns[-1].get("resolved_query", "") or self.turns[-1].get("query", "")

    def get_history_block(self) -> str:
        if not self.turns:
            return ""
        lines = ["--- CONVERSATION HISTORY ---"]
        for index, turn in enumerate(self.turns, start=1):
            lines.append(f"Q{index}: {turn['query']}")
            lines.append(f"A{index}: {turn['answer']}")
        lines.append("--- END HISTORY ---")
        return "\n".join(lines)

    def get_history_block_capped(self, max_tokens: int) -> str:
        """Return history trimmed so it never exceeds max_tokens tokens."""
        return _cap_history_block(self.get_history_block(), max_tokens)


class SessionConversationMemory:
    """DB-backed session memory using rolling summaries + recent turns."""

    def __init__(self, session_id: str, max_turns: int):
        self.session_id = session_id
        self.max_turns = max_turns

    @property
    def turns(self) -> list[dict[str, str]]:
        return list_session_turns(self.session_id)

    def add(
        self,
        query: str,
        answer: str,
        resolved_query: str | None = None,
        *,
        entities: dict | None = None,
        rendered_sources: list[dict] | None = None,
        primary_intent: str = "",
    ) -> None:
        # Derive turn_index from existing entity rows so it is always monotonically
        # increasing regardless of whether chat_messages is written by the caller.
        existing_entity_count = len(list_session_turn_entities(self.session_id, max_turns=10000))
        turn_index = existing_entity_count
        turns = self.turns + [
            {
                "query": query,
                "answer": answer,
                "resolved_query": resolved_query or query,
            }
        ]
        rolling_summary = ""
        if len(turns) > self.max_turns:
            older_turns = turns[:-self.max_turns]
            rolling_summary = _summarize_turns(older_turns)
        save_session_memory(
            self.session_id,
            rolling_summary=rolling_summary,
            last_resolved_query=(resolved_query or query).strip(),
        )
        # WS7: persist per-turn entity set.
        save_session_turn_entities(
            self.session_id,
            turn_index,
            primary_intent=primary_intent,
            original_query=query,
            resolved_query=(resolved_query or query),
            entities=entities or {},
        )
        del rendered_sources

    def latest_query(self) -> str:
        messages = list_session_messages(self.session_id)
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def latest_resolved_query(self) -> str:
        state = get_session_memory(self.session_id)
        if state["last_resolved_query"]:
            return state["last_resolved_query"]
        return self.latest_query()

    def recent_turn_entities(self, max_turns: int = 8) -> list[dict]:
        """Return last *max_turns* per-turn entity rows for this session."""
        return list_session_turn_entities(self.session_id, max_turns=max_turns)

    def latest_rendered_sources(self) -> list[dict]:
        message = latest_session_assistant_message(self.session_id)
        if not message:
            return []
        sources = message.get("sources") or []
        return list(sources) if isinstance(sources, list) else []

    def get_history_block(self) -> str:
        state = get_session_memory(self.session_id)
        recent_turns = self.turns[-self.max_turns :]
        if not state["rolling_summary"] and not recent_turns:
            return ""

        lines = []
        if state["rolling_summary"]:
            lines.append("--- CONVERSATION SUMMARY ---")
            lines.append(state["rolling_summary"])
            lines.append("--- END SUMMARY ---")
        if recent_turns:
            lines.append("--- CONVERSATION HISTORY ---")
            for index, turn in enumerate(recent_turns, start=1):
                lines.append(f"Q{index}: {turn['query']}")
                lines.append(f"A{index}: {turn['answer']}")
            lines.append("--- END HISTORY ---")
        return "\n".join(lines)

    def get_history_block_capped(self, max_tokens: int) -> str:
        """Return history trimmed so it never exceeds max_tokens tokens."""
        return _cap_history_block(self.get_history_block(), max_tokens)


class ThreadConversationMemory:
    """DB-backed thread memory using rolling summaries + recent turns."""

    def __init__(self, thread_id: str, session_id: str, max_turns: int):
        self.thread_id = thread_id
        self.session_id = session_id
        self.max_turns = max_turns

    @property
    def turns(self) -> list[dict[str, str]]:
        return list_thread_turns(self.thread_id)

    def add(
        self,
        query: str,
        answer: str,
        resolved_query: str | None = None,
        *,
        entities: dict | None = None,
        rendered_sources: list[dict] | None = None,
        primary_intent: str = "",
    ) -> None:
        # Derive turn_index from existing entity rows so it is always monotonically
        # increasing regardless of whether chat_messages is written by the caller.
        existing_entity_count = len(list_turn_entities(self.thread_id, max_turns=10000))
        turn_index = existing_entity_count
        turns = self.turns + [
            {
                "query": query,
                "answer": answer,
                "resolved_query": resolved_query or query,
            }
        ]
        rolling_summary = ""
        if len(turns) > self.max_turns:
            older_turns = turns[:-self.max_turns]
            rolling_summary = _summarize_turns(older_turns)
        save_thread_memory(
            self.thread_id,
            rolling_summary=rolling_summary,
            last_resolved_query=(resolved_query or query).strip(),
        )
        # WS7: persist per-turn entity set.
        save_turn_entities(
            self.thread_id,
            turn_index,
            primary_intent=primary_intent,
            original_query=query,
            resolved_query=(resolved_query or query),
            entities=entities or {},
        )
        del rendered_sources

    def latest_query(self) -> str:
        messages = list_thread_messages(self.thread_id)
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def latest_resolved_query(self) -> str:
        state = get_thread_memory(self.thread_id)
        if state["last_resolved_query"]:
            return state["last_resolved_query"]
        return self.latest_query()

    def recent_turn_entities(self, max_turns: int = 8) -> list[dict]:
        """Return last *max_turns* per-turn entity rows for this thread."""
        return list_turn_entities(self.thread_id, max_turns=max_turns)

    def latest_rendered_sources(self) -> list[dict]:
        message = latest_thread_assistant_message(self.thread_id)
        if not message:
            return []
        sources = message.get("sources") or []
        return list(sources) if isinstance(sources, list) else []

    def get_history_block(self) -> str:
        state = get_thread_memory(self.thread_id)
        recent_turns = self.turns[-self.max_turns :]
        if not state["rolling_summary"] and not recent_turns:
            return ""

        lines = []
        if state["rolling_summary"]:
            lines.append("--- CONVERSATION SUMMARY ---")
            lines.append(state["rolling_summary"])
            lines.append("--- END SUMMARY ---")
        if recent_turns:
            lines.append("--- CONVERSATION HISTORY ---")
            for index, turn in enumerate(recent_turns, start=1):
                lines.append(f"Q{index}: {turn['query']}")
                lines.append(f"A{index}: {turn['answer']}")
            lines.append("--- END HISTORY ---")
        return "\n".join(lines)

    def get_history_block_capped(self, max_tokens: int) -> str:
        """Return history trimmed so it never exceeds max_tokens tokens."""
        return _cap_history_block(self.get_history_block(), max_tokens)


def _summarize_turns(turns: list[dict[str, str]]) -> str:
    summary_lines = []
    for turn in turns[-12:]:
        query = " ".join(str(turn.get("query", "")).split())
        answer = " ".join(str(turn.get("answer", "")).split())
        if len(answer) > 220:
            answer = answer[:217].rstrip() + "..."
        summary_lines.append(f"- Q: {query}\n  A: {answer}")
    return "\n".join(summary_lines)
