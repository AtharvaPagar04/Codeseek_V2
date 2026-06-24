# Manual Regression Evaluation Policy

CodeSeek maintains a curated, high-priority **15-query manual regression set** to benchmark the accuracy of the retrieval layer, fusion (RRF), multi-turn conversation parsing, and intent classification.

## Purpose of the 15-Query Set

This regression suite contains critical query patterns covering common developer usage profiles:
- **Simple Repository Overview**: Queries asking about project scope, directory structure, or backend module mappings.
- **Precise Code Snippet Extraction**: Queries demanding specific functional boundaries (e.g., auth checks, safe eval runners, Qdrant upserts).
- **Source Location Identification**: Direct requests asking where specific subsystems or configurations are implemented.
- **Technical/Architecture Explanation**: Demands for architectural walkthroughs of the retrieval pipelines.
- **Multi-Turn & Follow-Up**: Multi-turn sequences to verify topic tracking, follow-up intent resolution, and context isolation.

### The 15 Regression Queries

1. **What is this project about?** (Category: *Overview*)
2. **How is this codebase structured?** (Category: *Overview*)
3. **What are the main backend modules?** (Category: *Overview*)
4. **show me _require_auth code** (Category: *Code Snippet*)
5. **provide me the auth function code** (Category: *Code Snippet*)
6. **show me the safe eval runner code** (Category: *Code Snippet*)
7. **show me the evaluation report API endpoint code** (Category: *Code Snippet*)
8. **show me the Qdrant upsert code** (Category: *Code Snippet*)
9. **Where is safe eval implemented?** (Category: *Source Location*)
10. **Where is evaluation report API implemented?** (Category: *Source Location*)
11. **show me safe eval docs** (Category: *Code Snippet*)
12. **How does the retrieval pipeline work?** (Category: *Technical Explanation*)
13. **Where is reranking handled in searcher.py?** (Category: *Source Location*)
14. **show me the Qdrant upsert code -> explain that** (Category: *Multi-turn / Follow-up*)
15. **show me the safe eval runner code -> explain that** (Category: *Multi-turn / Follow-up*)

---

## Execution and Automation Policy

> [!IMPORTANT]
> The manual regression set is **manually triggered or explicitly run** by developers during evaluation cycles. It is **never automatically run** during normal application usage or chat sessions to prevent high latency, token consumption, and unnecessary local GPU resource load.

### Triggering Regression Evaluators
To run regression evaluations locally:
1. Ensure your active development session is fully indexed and ready.
2. Run the focused regression test suites via the python environment:
   ```bash
   cd backend
   # To validate retrieval-specific expectations
   .venv/bin/pytest tests/test_retrieval_source_filtering.py
   
   # To validate conversational multi-turn context
   .venv/bin/pytest tests/test_code_snippet_answer.py
   ```
3. Use the global safe evaluation script to inspect full regression logs:
   ```bash
   .venv/bin/python evals/run_safe_evals.py --session-id <session-id>
   ```
