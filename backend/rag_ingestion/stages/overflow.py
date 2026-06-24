"""Overflow handling stage."""

from dataclasses import replace

from rag_ingestion.config import MAX_CHUNK_TOKENS, SLIDING_OVERLAP, SLIDING_WINDOW_SIZE
from rag_ingestion.models.chunk import Chunk


def handle_overflow(chunks: list[Chunk]) -> list[Chunk]:
    """Split oversized chunks using a sliding line window."""
    expanded: list[Chunk] = []

    for chunk in chunks:
        token_count = _count_tokens(chunk.content)

        if token_count <= MAX_CHUNK_TOKENS:
            chunk.token_count = token_count
            chunk.chunk_part = 1
            chunk.total_parts = 1
            expanded.append(chunk)
            continue

        windows = _line_windows(chunk.content)
        total_parts = len(windows)

        for index, (content, relative_start, relative_end) in enumerate(
            windows, start=1
        ):
            expanded.append(
                replace(
                    chunk,
                    content=content,
                    start_line=chunk.start_line + relative_start - 1,
                    end_line=chunk.start_line + relative_end - 1,
                    chunk_part=index,
                    total_parts=total_parts,
                    token_count=_count_tokens(content),
                )
            )

    return expanded


def _count_tokens(content: str) -> int:
    import tiktoken

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline fallback for test environments
        class _FallbackEncoding:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8", errors="ignore")

        encoding = _FallbackEncoding()

    return len(encoding.encode(content))


def _line_windows(content: str) -> list[tuple[str, int, int]]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return [("", 1, 1)]

    step = max(1, SLIDING_WINDOW_SIZE - SLIDING_OVERLAP)
    windows: list[tuple[str, int, int]] = []

    for start in range(0, len(lines), step):
        window = lines[start : start + SLIDING_WINDOW_SIZE]
        if not window:
            break

        relative_start = start + 1
        relative_end = start + len(window)

        windows.append(("".join(window), relative_start, relative_end))

        if start + SLIDING_WINDOW_SIZE >= len(lines):
            break

    return windows