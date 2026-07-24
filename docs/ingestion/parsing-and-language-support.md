# Parsing and Language Support

## AST Parsing

Tree-sitter parsing is implemented for:

| Language | Extensions |
|---|---|
| Python | `.py` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.tsx` |

These parsers extract imports and source symbols. A parser failure falls back to a file-level record with no extracted symbols or imports.

## File-Level Content

The pipeline also accepts Markdown, MDX, JSON, TOML, YAML, text, HTML, CSS, shell, `.conf`, Dockerfile, Caddyfile, `.gitignore`, and environment example files. These are indexed without Tree-sitter symbol extraction.

Known frontend configuration files, including Vite, Next.js, Tailwind, ESLint, Jest, Rollup, Vitest, and TypeScript configurations, are deliberately treated as file-level documents.

## Filtering

Discovery walks all files, then applies:

1. Repository `.gitignore` rules.
2. Built-in ignored directories such as `.git`, `node_modules`, build outputs, caches, and virtual environments.
3. Built-in ignored filenames, extensions, and generated-file patterns.

Secret-bearing `.env` variants are ignored. `.env.example` and matching example variants are supported. Lockfiles, databases, archives, binaries, media, logs, source maps, and generated bundles are excluded.

Files with no supported language or file type are marked `unsupported_language` and are not chunked.
