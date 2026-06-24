# CodeSeek Evaluation Policy and Gating Rules (v1)

This document defines the formal gating policy for CodeSeek's evaluation suites. It classifies all evaluation findings into one of three tiers: **Hard Gates**, **Soft Warnings**, and **Diagnostic-only Observations**.

This policy layer protects our deployment and validation pipeline by separating clear regressions (which fail the build/run) from softer signals that should be reviewed without blocking release by default.

---

## 1. Finding Classification Summary

### Hard Gates (FAIL or ERROR)
These conditions represent direct functionality regressions or clear evaluation failures. Any hard gate failure results in an overall status of **ERROR** for the run.
* **Retrieval Eval status is FAIL or ERROR**: Any regression in deterministic retrieval metrics (e.g. `file_hit@5` dropping below the configured gate) is blocked.
* **Conversation Eval status is FAIL or ERROR**: Multi-turn followup context mapping or intent classification failure.
* **Exact Hit Regressions > 0**: A query that previously hit the target at a high rank now fails to do so.
* **Protected Hit Preservation < 100%**: Any drop in the retrieval of explicitly protected chunks when protected hits exist.
* **Empty Result Rate > 0**: Unless explicitly configured via `--allow-empty-results`, any query returning zero retrieved contexts is blocked.

### Soft Warnings (WARN)
These conditions represent potential degradations or minor quality regressions. They do not block the build or run, but they generate a status of **WARN** to alert reviewers.
* No soft warnings are emitted by the current policy runner. Add them only when a deterministic signal is stable enough to be actionable.

### Diagnostic-only Observations (PASS)
These observations are kept purely for trend tracking and research. They **must not** change a run's status to WARN or ERROR.
* No diagnostic-only observations are emitted by the current policy runner.

---

## 2. Policy Verification
Evaluation runs are aggregated and gated automatically using `eval_policy_summary.py`. This script reads the JSON reports from retrieval and conversation evaluations, evaluates them against the criteria defined above, and generates a structured status report.

For a unified command-line tool that orchestrates this entire pipeline sequentially and handles gates, see the [Safe Eval Runner](safe_eval_runner.md) documentation.
