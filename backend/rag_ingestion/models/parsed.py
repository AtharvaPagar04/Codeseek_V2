"""Parsed AST output models."""

from dataclasses import dataclass, field


@dataclass
class ParsedSymbol:
    """A symbol extracted from a source file."""

    symbol_name: str
    symbol_type: str
    parent_symbol: str
    start_line: int
    end_line: int
    parameters: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    signature: str = ""
    docstring: str = ""
    calls: list[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    """Parser output for one source file."""

    relative_path: str
    language: str
    parse_status: str
    imports: list[str] = field(default_factory=list)
    symbols: list[ParsedSymbol] = field(default_factory=list)
