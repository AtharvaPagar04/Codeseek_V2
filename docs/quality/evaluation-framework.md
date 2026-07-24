# Evaluation Framework

CodeSeek includes two evaluation systems under `backend/`.

## Dataset Runner

`backend/scripts/retrieval_eval.py` runs JSON datasets from `backend/evals/datasets/`. It reports retrieval, source, answer, memory, refusal, grounding, and latency metrics, including Hit@K, MRR@K, citation coverage, expected file and symbol matches, follow-up precision/recall, and source faithfulness.

```bash
cd backend
PYTHONPATH=. .venv/bin/python scripts/retrieval_eval.py \
  --eval-file evals/datasets/eval_codeseek_exact_wording.json --k 10
```

`retrieval_eval_suite.py` executes the configured multi-repository suite and aggregates weighted metrics.

## Golden Evaluation

`backend/evals/retrieval_eval.py` evaluates a live session against YAML golden queries. `conversation_eval.py` evaluates multi-turn trees. The safe runner verifies the expected session, repository root, and collection before running retrieval, conversation, and policy-gating reports.

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m evals.run_safe_evals \
  --session-id <id> \
  --expected-repo-root <path> \
  --expected-collection <collection> \
  --output-dir evals/reports/safe_eval_run
```

## Index Health

`evals/index_health.py` validates collection contents, session metadata, file/chunk mappings, and freshness before evaluation. Failed checks produce full-reindex guidance.

Use a newly indexed session for final retrieval validation so stored chunks match the current ingestion and metadata code.
