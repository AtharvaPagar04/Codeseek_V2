"""Lightweight secret scanner for tracked text files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS: dict[str, re.Pattern[str]] = {
    "groq_api_key": re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_pat": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "generic_assignment": re.compile(
        r"(?im)^\s*[A-Za-z_]*?(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{16,}['\"]\s*$"
    ),
}

SKIP_FILES = {
    ".env",
    ".env.example",
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".lock",
    ".pyc",
}


def _tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    files = []
    for line in out.splitlines():
        path = Path(line)
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def main() -> int:
    hits: list[str] = []
    for path in _tracked_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                hits.append(f"{path}:{line} [{name}]")

    if hits:
        print("Potential secrets detected:")
        for hit in hits:
            print(f"- {hit}")
        print("Failing build. Remove/rotate secrets and retry.")
        return 1

    print("No potential secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
