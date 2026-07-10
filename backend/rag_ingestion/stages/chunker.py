"""Chunk generation stage."""

import re
from pathlib import Path

from rag_ingestion.models.chunk import Chunk
from rag_ingestion.models.file import FileRecord
from rag_ingestion.models.parsed import ParsedFile
from retrieval.search.source_truth import analyze_source_truth


def _extract_imported_symbols(imports: list[str]) -> list[str]:
    names: list[str] = []
    for statement in imports or []:
        match = re.search(r'import\s+\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']', statement)
        if match:
            for part in match.group(1).split(","):
                cleaned = part.strip()
                if not cleaned:
                    continue
                names.append(cleaned.split(" as ", 1)[0].strip())
            continue

        mixed_match = re.search(
            r'import\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*,\s*\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']',
            statement,
        )
        if mixed_match:
            names.append(mixed_match.group(1).strip())
            for part in mixed_match.group(2).split(","):
                cleaned = part.strip()
                if cleaned:
                    names.append(cleaned.split(" as ", 1)[0].strip())
            continue

        default_match = re.search(
            r'import\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+from\s+["\']([^"\']+)["\']',
            statement,
        )
        if default_match:
            names.append(default_match.group(1).strip())
            continue

        ns_match = re.search(
            r'import\s+\*\s+as\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+from\s+["\']([^"\']+)["\']',
            statement,
        )
        if ns_match:
            names.append(ns_match.group(1).strip())
            continue

        py_match = re.match(r"^from\s+([.\w]+)\s+import\s+(.+)$", statement.strip())
        if py_match:
            for part in py_match.group(2).strip().strip("()").split(","):
                cleaned = part.strip()
                if cleaned and cleaned != "*":
                    names.append(cleaned.split(" as ", 1)[0].strip())

    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _extract_used_symbols(content: str) -> list[str]:
    names = re.findall(r"<([A-Z][A-Za-z0-9_]*)\b", content or "")
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def generate_chunks(parsed: ParsedFile, file: FileRecord) -> list[Chunk]:
    """Convert parser output into source chunks."""
    lines = Path(file.path).read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )

    if parsed.parse_status == "failed":
        return [
            Chunk(
                file_path=file.path,
                relative_path=file.relative_path,
                language=file.language,
                chunk_type="file",
                start_line=1 if lines else 0,
                end_line=len(lines),
                imports=parsed.imports,
                content="".join(lines),
            )
        ]

    chunks: list[Chunk] = []
    file_symbols = [symbol.symbol_name for symbol in parsed.symbols]
    imported_symbols = _extract_imported_symbols(parsed.imports)
    file_content = "".join(lines)
    used_symbols = _extract_used_symbols(file_content)
    source_truth = analyze_source_truth(
        relative_path=file.relative_path,
        content=file_content,
        imports=parsed.imports,
        exported_symbols=file_symbols,
    )

    # Always include a file-level chunk to cover module-level imports, comments, docstrings, and global definitions.
    chunks.append(
        Chunk(
            file_path=file.path,
            relative_path=file.relative_path,
            language=file.language,
            chunk_type="file",
            start_line=1 if lines else 0,
            end_line=len(lines),
            imports=parsed.imports,
            file_symbols=file_symbols,
            symbol_role="definition" if file_symbols else ("import" if imported_symbols else ""),
            defined_symbols=file_symbols,
            used_symbols=used_symbols,
            imported_symbols=imported_symbols,
            source_of_truth=bool(source_truth["source_of_truth"]),
            centrality_score=float(source_truth["centrality_score"]),
            exported_symbols=list(source_truth["exported_symbols"]),
            content=file_content,
        )
    )

    for symbol in parsed.symbols:
        content = "".join(lines[symbol.start_line - 1 : symbol.end_line])
        symbol_used = _extract_used_symbols(content)
        chunks.append(
            Chunk(
                file_path=file.path,
                relative_path=file.relative_path,
                language=file.language,
                chunk_type=symbol.symbol_type,
                symbol_name=symbol.symbol_name,
                parent_symbol=symbol.parent_symbol,
                signature=symbol.signature,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                imports=parsed.imports,
                calls=symbol.calls,
                parameters=symbol.parameters,
                methods=symbol.methods,
                symbol_role="definition",
                defined_symbols=[symbol.symbol_name],
                used_symbols=symbol_used,
                imported_symbols=imported_symbols,
                source_of_truth=False,
                centrality_score=0.0,
                exported_symbols=[],
                docstring=symbol.docstring,
                content=content,
            )
        )

    return chunks
