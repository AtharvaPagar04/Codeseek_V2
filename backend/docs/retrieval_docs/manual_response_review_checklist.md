# CodeSeek: Manual Response-Review Checklist

This checklist is used to manually audit CodeSeek's retrieval-augmented generation (RAG) answers. While automated evaluations check files, symbols, hit@k, and latency, manual audits ensure **grounding accuracy, formatting style compliance, and overall usefulness**.

Reviewers should sample 10–20 queries per release and grade them using this rubric.

---

## 1. Grounding and Context Discipline (Safety)

Grounded answers must only contain facts and code structures visible in the retrieved code snippets (context).

- [ ] **No Hallucinated Symbols/Files**: Ensure every class, method, function, variable name, and file path mentioned in the answer actually exists in the provided context.
- [ ] **Strict Allowed Sources Constraint**: Did the model only cite files/symbols listed under the `ALLOWED SOURCES` section?
- [ ] **No Out-of-Context Claims**: Verify that the description of what the code does, reads, writes, or calls is directly supported by the context. No external general knowledge about libraries or frameworks should be introduced if it isn't visible in the codebase context.
- [ ] **Safe Absence/Negative Answers**: For queries where evidence is absent, did the model avoid absolute repository-wide claims (e.g. "This repo does not support X") and instead state "Not found in retrieved context" with helpful next steps?

---

## 2. Answer Usefulness & Tone (Quality)

Answers should be direct, actionable, and structured for developers.

- [ ] **Response Mode Alignment**: Did the orchestrator select the correct response mode based on query intent?
  - `CODE REQUEST`: Verbatim code snippets, minimal prose.
  - `OVERVIEW`: High-level explanation of tech stack, module layout, and runtime shape.
  - `EXPLANATION`: Mid-level walk-through of ui/backend components, interactions, and data transformations.
  - `TECHNICAL TRACE`: Step-by-step numbered/bulleted request-response path or method call sequence.
- [ ] **Grounded Technical Walk-through**: For traces or deep-dives, does the answer trace the exact inputs, logic steps, return values, side effects, and connections between components instead of offering generic summaries?
- [ ] **Code Snippet Gating**: Are fenced code blocks minimized? They must only appear if the user explicitly asked to "show/write/provide code" or if a snippet is the cleanest way to explain syntax.
- [ ] **Verbatim Excerpts**: When snippets are shown, are they exact verbatim lines from the context? Ensure the file and symbol names are written clearly above each block.

---

## 3. Formatting, References, & Style (Aesthetics)

CodeSeek answers must follow strict visual and formatting rules to ensure a premium developer experience.

- [ ] **Inline References**: Check that inline code styling is used for all symbols (e.g. `` `file.py :: ClassName.method` ``) instead of plain text or excessive block quotes.
- [ ] **Default Structural Template**: Unless overridden by a specific response mode, does the answer follow the standard template?
  1. A one-line direct answer.
  2. 3–6 concise, evidence-backed bullet points.
- [ ] **No Duplicate Sections**: Verify that there are no duplicate "Key evidence" or "Sources" sections in the prose. Citations are handled by the UI source cards.
- [ ] **Clean Typography**: Ensure there are no broken markdown tags, unmatched backticks, or misaligned list indentations.

---

## 4. Fallbacks & Evidence Banners

When context is low, partial, or weak, CodeSeek must guide the user actionably.

- [ ] **Low-Context Fallback**: If `shown_sources` was empty, did it return the exact `LOW_CONTEXT_FALLBACK` string?
- [ ] **Warning Banner Accuracy**: Did the response header contain the correct evidence banner?
  - `PARTIAL_EVIDENCE_BANNER`: Shown for partial evidence (1 source, low overlap score).
  - `WEAK_EVIDENCE_BANNER`: Shown for weak/unreliable evidence.
- [ ] **Actionable Guidance**: Do fallback/warning messages suggest specific alternative actions? They should prompt the user to search using exact symbol/file names, target a different query mode, or refine search queries rather than just stating "confidence is low".

---

## 5. Reviewer Scorecard & Evaluation Rubric

For each query audited, assign a score using the following criteria:

| Dimension | Criteria | Max Points | Score |
| :--- | :--- | :---: | :---: |
| **Grounding** | **3**: 100% grounded, no hallucinations.<br>**2**: Minor extrapolation (unsupported detail but correct code).<br>**1**: Hallucinated symbol/file.<br>**0**: Fabricated/hallucinated answer body. | 3 | |
| **Usefulness** | **3**: Direct, complete, answers the developer's question.<br>**2**: Answers the question but lacks technical depth or connection.<br>**1**: Vague or misses the main query intent.<br>**0**: Useless/misleading. | 3 | |
| **Formatting** | **2**: Adheres perfectly to template, correct inline citations, proper code block usage.<br>**1**: Minor format violations (too many bullets, code blocks when not requested).<br>**0**: Messy formatting / no structure. | 2 | |
| **Banners** | **2**: Correct fallback/confidence banner shown, actionable rephrasing advice.<br>**1**: Banner mismatch or non-actionable fallback.<br>**0**: No banner on weak context. | 2 | |
| **Total** | **Pass Threshold: $\ge$ 8 / 10** | **10** | |

### Sample Audit Table

| Case ID | Query | Actual Response Mode | Grounding (3) | Usefulness (3) | Formatting (2) | Banners (2) | Total (10) | Pass/Fail | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `audit-001` | | | | | | | | | |
| `audit-002` | | | | | | | | | |
