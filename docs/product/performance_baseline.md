# CodeSeek Performance Baseline V1

This document outlines the performance metrics that matter for CodeSeek, how to run and interpret the benchmark script, and best practices for establishing and recording baseline results safely on a development machine.

---

## 1. Metrics That Matter

To ensure CodeSeek remains responsive, lightweight, and efficient, we track four core performance categories:

1. **Backend Health Latency**: The round-trip time of a lightweight `/api/v1/health` request. This measures core API gateway overhead independent of DB/LLM/Vector queries.
2. **Query Latency**: The wall-clock time required to retrieve search results, build context, prompt the LLM, and parse the output response.
3. **Full vs. Incremental Indexing Duration**:
   - **Full Indexing**: The total time to parse, chunk, embed, and store an entire repository.
   - **Incremental Indexing**: The time to detect modified files and selectively update only the modified chunks/embeddings.
4. **Frontend Build Size**: The aggregated size of all production build assets under `frontend/dist`. Smaller build sizes guarantee fast initial load times in the browser.

---

## 2. Benchmark Script Usage

The lightweight script `scripts/perf_baseline.sh` provides both safe, non-destructive measurements (default) and active, target-driven benchmarking (via flags).

### Safe Default Mode
By default, the script executes instantly without mutating data, making external requests, or requiring authorization secrets:
```bash
./scripts/perf_baseline.sh
```
This measures and outputs:
- Latency of the local backend service (if online).
- Total file size of compiled production frontend assets.
- Latest indexing job summary and duration directly from DB history.

### Active Benchmarking Modes
To run active latency or indexing benchmarks, use the explicit, opt-in flags:

- **Measure Query Latency**:
  ```bash
  ./scripts/perf_baseline.sh --run-query
  ```
  Fires a sample query request against the first session in the database and measures response time.

- **Measure Full Indexing**:
  ```bash
  ./scripts/perf_baseline.sh --run-index
  ```
  Triggers a background full re-index job on the latest session and polls it until completion, tracking the active duration.

- **Measure Incremental Indexing**:
  ```bash
  ./scripts/perf_baseline.sh --run-incremental
  ```
  Triggers a background incremental re-index job on the latest session and polls it until completion.

- **Dry Run**:
  ```bash
  ./scripts/perf_baseline.sh --dry-run --run-query --run-index
  ```
  Prints out the steps and executes the flow without performing actual network/indexing mutations.

---

## 3. Laptop-Safe Benchmarking Rules

To get reproducible results and avoid thermal throttling or resource starvation on development environments:

1. **Close Resource-Intensive Apps**: Close IDEs, Slack, video tools, and extra browser tabs before starting a benchmark.
2. **Observe Cooldown Periods**: Allow the CPU to cool down for at least 15-30 seconds between successive indexing runs.
3. **Ensure Plugged-In Power**: Perform measurements with the laptop connected to a power outlet and set to "High Performance" power mode.
4. **Monitor System Temp**: If the fan starts spinning at maximum speed, pause benchmarks to prevent CPU throttling which artificially inflates results.

---

## 4. How to Record Baseline Results Manually

When optimizing or introducing changes, record your metrics in a local markdown log or commit message using the following template:

```markdown
### CodeSeek Baseline Run
- Date: YYYY-MM-DD
- CPU/Machine Spec: (e.g., Apple M2 Pro, Intel i7-13700H)
- Memory: (e.g., 16 GB)

| Metric | Baseline Value | Optimized Value | Change % |
| --- | --- | --- | --- |
| Health Latency | 2.4 ms | - | - |
| Frontend Size | 0.50 MB | - | - |
| Full Index Time | 45.2 s | - | - |
| Incremental Index | 2.1 s | - | - |
| Query Latency | 850 ms | - | - |
```
