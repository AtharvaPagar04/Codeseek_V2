# Chat Diagnostics Panel

CodeSeek includes a per-answer diagnostics panel designed to provide developer-level visibility into how a response was retrieved, parsed, and validated. The diagnostics panel is collapsed by default and can be expanded for debugging search behaviors, prompt intent classification, and source tracking.

---

## Diagnostics Panel Sections

The panel organizes information into five primary categories:

### 1. Intent
This section captures the semantic intent analysis of your question:
* **Intent:** The classified query type (e.g., `CODE_REQUEST`, `DOCUMENTATION_REQUEST`, `GENERAL_CONVERSATION`).
* **Primary Intent:** Under-the-hood intent label mapping (advanced).
* **Response Mode:** Mode selected for the output structure (e.g., `code_snippet`, `summary`).
* **Routing Mode:** The execution route taken (e.g., `local` LLM or `remote` API).

### 2. Model
This section displays details about the active language model used to generate the answer:
* **Model:** The active model name (e.g., `local / qwen2.5-coder:3b-8k` or `openai / gpt-4o`).
* **Context Tokens:** The volume of prompt tokens sent to the model (advanced).

### 3. Sources
This section tracks what files and code segments were retrieved:
* **Evidence Confidence:** The system's assessment of how well the retrieved code matches your query (e.g., `strong`, `moderate`, `weak`).
* **Source Filter:** Active restrictions applied to search paths (advanced).
* **Selected Sources:** List of source files used as context (advanced).
* **Reasoning Sources:** List of intermediate files analysed for context (advanced).
* **Rendered Sources:** Files shown to the user (advanced).

### 4. Validation
This section provides details about answer evaluation:
* **Validation:** Whether the generated response passed consistency audits (e.g., confirming retrieved file paths exist in the answer and formatting rules are met).
* **Repaired:** Indicates if the response had to be corrected/repaired by the system due to validation failures.

### 5. Freshness
This section exposes Git integration status for the session:
* **Session Status:** The status of the workspace index (e.g., `ready`, `failed`).
* **Freshness Status:** The synchronization level (e.g., `up_to_date`, `dirty_worktree`, `out_of_date`).
* **Indexed branch:** The branch that was parsed (advanced).
* **Indexed Commit / Current Commit:** Commit SHAs (advanced).
* **Dirty Worktree:** Indicates if local changes are present (advanced).
* **Last Checked:** Timestamp of the last Git status check (advanced).

---

## Basic vs. Advanced Fields

To prevent UI clutter, CodeSeek separates basic overview fields from deep-dive metrics:
* **Basic Fields:** Displayed immediately when you expand the main Diagnostics section. These cover high-level statuses like active Model, Intent classification, Evidence confidence, and Validation.
* **Advanced Fields:** Tucked inside a nested, collapsible **Advanced details** panel within each category. These hide granular details (such as commit SHAs, file lists, and token counts) until explicitly requested.

---

## Copy Diagnostics Utility

In the diagnostics panel header, a **Copy** button is available. Clicking this button:
1. Gathers all diagnostic metadata (both basic and advanced).
2. Formats the data into a clean, human-readable Markdown structure.
3. Copies it to your clipboard.

This makes it easy to paste diagnostic records into issue trackers, documentation, or debugging conversations.

---

## Intentionally Hidden Fields (Security & Secrecy)

To ensure privacy and security, CodeSeek enforces strict filtering on the diagnostics payload. The following data is **never** exposed to the user interface or client-side diagnostics panel:
* **API Keys & Credentials:** Custom keys for providers like OpenAI or Anthropic are encrypted and never returned in plaintext.
* **Auth Tokens:** Session identifiers, GitHub OAuth tokens, and system authentication cookies are completely stripped.
* **Raw Prompts:** The exact prompt templates, system instructions, and engineering templates sent to LLMs are hidden.
* **Hidden Retrieval Payloads:** Unused database context, database indices, and system files are excluded.
* **Environment Secrets:** Internal environment configurations and server-side secret keys are hidden.
* **Raw Provider Payloads:** Full, unfiltered raw JSON responses returned by LLM APIs.
