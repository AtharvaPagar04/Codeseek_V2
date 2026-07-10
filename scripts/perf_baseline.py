# [ignoring loop detection]
import os
import sys
import time
import argparse
import urllib.request
import json
from datetime import datetime

# Helper to load backend/.env file properties
def load_env_properties(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                k, v = stripped.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Find workspace root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_env_properties(os.path.join(root_dir, "backend", ".env"))

# Resolve python package import paths
sys.path.append(os.path.join(root_dir, "backend"))

# Parse command line options
cli_parser = argparse.ArgumentParser(description="CodeSeek Performance Baseline Runner")
cli_parser.add_argument("--run-query", action="store_true", help="Execute sample query and measure latency")
cli_parser.add_argument("--run-index", action="store_true", help="Trigger and measure full indexing duration")
cli_parser.add_argument("--run-incremental", action="store_true", help="Trigger and measure incremental indexing duration")
cli_parser.add_argument("--dry-run", action="store_true", help="Perform a dry run of the benchmark actions")
cmd_args = cli_parser.parse_args()

# Terminal style codes
CLR_GREEN = "\033[0;32m"
CLR_YELLOW = "\033[1;33m"
CLR_RED = "\033[0;31m"
CLR_BLUE = "\033[0;34m"
CLR_RESET = "\033[0m"

print(f"{CLR_BLUE}======================================================================{CLR_RESET}")
print(f"{CLR_BLUE}                CodeSeek Performance Baseline V1                      {CLR_RESET}")
print(f"{CLR_BLUE}======================================================================{CLR_RESET}\n")

# 1. Backend Health Check Latency
backend_url = os.getenv("CODESEEK_BACKEND_URL", "http://localhost:8000")
health_endpoint = f"{backend_url}/api/v1/health"

print("--- 1. Backend Health Latency ---")
health_status = "Offline"
health_latency = "N/A"

start_t = time.perf_counter()
try:
    if cmd_args.dry_run:
        health_status = "Online (Dry-run)"
        health_latency = "0.0 ms"
    else:
        req = urllib.request.Request(health_endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                elapsed = (time.perf_counter() - start_t) * 1000
                health_status = "Online"
                health_latency = f"{elapsed:.2f} ms"
except Exception as e:
    health_status = f"Offline / Error ({e})"

print(f"Status:  {health_status}")
print(f"Latency: {health_latency}\n")

# 2. Frontend Build Size
print("--- 2. Frontend Asset Size ---")
dist_dir = os.path.join(root_dir, "frontend", "dist")
dist_size_str = "N/A (Run 'npm run build' inside frontend/ to measure)"

if os.path.exists(dist_dir):
    total_bytes = 0
    file_count = 0
    for root, dirs, files in os.walk(dist_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
                file_count += 1
    dist_size_str = f"{total_bytes / (1024 * 1024):.2f} MB ({file_count} files)"

print(f"Build folder: {dist_dir}")
print(f"Total size:   {dist_size_str}\n")

# 3. Database Jobs History
print("--- 3. Latest Indexing Job (History) ---")
latest_job_str = "No repository sessions or jobs found"
latest_job_details = None

try:
    from retrieval.db import db_cursor, list_indexing_jobs
    with db_cursor() as (conn, cursor):
        row = cursor.execute("SELECT id, repo_full_name FROM repo_sessions LIMIT 1").fetchone()
        if row:
            session_id = row["id"]
            repo_name = row["repo_full_name"]
            jobs = list_indexing_jobs(session_id, limit=1)
            if jobs:
                job = jobs[0]
                latest_job_details = job
                
                # Parse duration
                duration_str = "N/A"
                if job.get("started_at") and job.get("completed_at"):
                    try:
                        # SQLite stores ISO strings
                        fmt = "%Y-%m-%dT%H:%M:%S.%f"
                        # Handle potential timezone offsets
                        start_str = job["started_at"].split("+")[0]
                        comp_str = job["completed_at"].split("+")[0]
                        t_start = datetime.strptime(start_str, fmt)
                        t_comp = datetime.strptime(comp_str, fmt)
                        diff = (t_comp - t_start).total_seconds()
                        duration_str = f"{diff:.1f} seconds"
                    except Exception:
                        duration_str = "Completed (unknown duration)"
                elif job.get("started_at"):
                    duration_str = "Running / Incomplete"
                
                latest_job_str = (
                    f"Job ID:          {job['job_id']}\n"
                    f"Session ID:      {job['session_id']} ({repo_name})\n"
                    f"Mode:            {job['indexing_mode'].upper()}\n"
                    f"Status:          {job['status'].upper()}\n"
                    f"Duration:        {duration_str}\n"
                    f"Files Indexed:   {job.get('files_indexed', 0)}\n"
                    f"Chunks:          {job.get('chunks_generated', 0)}\n"
                    f"Embeddings:      {job.get('embeddings_stored', 0)}"
                )
            else:
                latest_job_str = f"Session {session_id} ({repo_name}) has no indexing jobs."
        else:
            latest_job_str = "No repo sessions in database."
except Exception as db_err:
    latest_job_str = f"Database read failure: {db_err}"

print(f"{latest_job_str}\n")

# 4. Optional query measurement
query_latency_str = "SKIPPED (Pass --run-query to execute)"
if cmd_args.run_query:
    print("--- 4. Query Latency ---")
    if cmd_args.dry_run:
        query_latency_str = "0.0 ms (Dry-run)"
        print(f"Result: {query_latency_str}\n")
    else:
        # Resolve target session id
        target_session = None
        try:
            from retrieval.db import db_cursor
            with db_cursor() as (conn, cursor):
                row = cursor.execute("SELECT id FROM repo_sessions LIMIT 1").fetchone()
                if row:
                    target_session = row["id"]
        except Exception:
            pass

        if not target_session:
            query_latency_str = "ERROR: No session found in DB to query"
            print(f"Result: {query_latency_str}\n")
        else:
            api_key = os.getenv("CODESEEK_API_KEY", "")
            query_payload = {"session_id": target_session, "query": "Find performance benchmark baseline sample"}
            query_headers = {"Content-Type": "application/json"}
            if api_key:
                query_headers["Authorization"] = f"Bearer {api_key}"

            query_url = f"{backend_url}/api/v1/query"
            query_start = time.perf_counter()
            try:
                data_bytes = json.dumps(query_payload).encode("utf-8")
                req = urllib.request.Request(query_url, data=data_bytes, headers=query_headers, method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status == 200:
                        query_diff = (time.perf_counter() - query_start) * 1000
                        query_latency_str = f"{query_diff:.2f} ms"
                    else:
                        query_latency_str = f"Error (HTTP {resp.status})"
            except Exception as q_exc:
                query_latency_str = f"Error ({q_exc})"
            print(f"Result: {query_latency_str}\n")

# 5. Optional full or incremental index measurement
index_latency_str = "SKIPPED (Pass --run-index or --run-incremental to execute)"
if cmd_args.run_index or cmd_args.run_incremental:
    print("--- 5. Active Indexing Duration ---")
    if cmd_args.dry_run:
        index_latency_str = "0.0 seconds (Dry-run)"
        print(f"Result: {index_latency_str}\n")
    else:
        target_session = None
        try:
            from retrieval.db import db_cursor
            with db_cursor() as (conn, cursor):
                row = cursor.execute("SELECT id FROM repo_sessions LIMIT 1").fetchone()
                if row:
                    target_session = row["id"]
        except Exception:
            pass

        if not target_session:
            index_latency_str = "ERROR: No session found in DB to index"
            print(f"Result: {index_latency_str}\n")
        else:
            api_key = os.getenv("CODESEEK_API_KEY", "")
            action_headers = {}
            if api_key:
                action_headers["Authorization"] = f"Bearer {api_key}"

            mode = "full" if cmd_args.run_index else "incremental"
            endpoint_suffix = "index-latest" if mode == "full" else "index-incremental"
            action_url = f"{backend_url}/api/v1/sessions/{target_session}/{endpoint_suffix}"

            print(f"Triggering {mode.upper()} indexing on session {target_session}...")
            job_id = None
            try:
                req = urllib.request.Request(action_url, headers=action_headers, method="POST")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        job_id = resp_data.get("job_id")
                        print(f"Job triggered successfully. Job ID: {job_id}")
                    else:
                        index_latency_str = f"Trigger Error (HTTP {resp.status})"
            except Exception as trigger_exc:
                index_latency_str = f"Trigger Error ({trigger_exc})"

            if job_id:
                print("Polling job status...")
                from retrieval.db import list_indexing_jobs
                start_poll = time.perf_counter()
                completed = False
                while not completed:
                    time.sleep(1.0)
                    try:
                        jobs = list_indexing_jobs(target_session, limit=20)
                        matched = [j for j in jobs if j["job_id"] == job_id]
                        if matched:
                            j = matched[0]
                            status = j["status"]
                            print(f"Current stage: {j.get('current_stage')} | Status: {status}")
                            if status in ("succeeded", "failed", "cancelled"):
                                completed = True
                                diff_s = time.perf_counter() - start_poll
                                index_latency_str = (
                                    f"{diff_s:.1f} seconds ({status.upper()})\n"
                                    f"  Files Indexed:   {j.get('files_indexed', 0)}\n"
                                    f"  Chunks:          {j.get('chunks_generated', 0)}\n"
                                    f"  Embeddings:      {j.get('embeddings_stored', 0)}"
                                )
                                if j.get("error"):
                                    index_latency_str += f"\n  Error:           {j['error']}"
                        else:
                            # If job is not in list yet or deleted
                            pass
                    except Exception as poll_exc:
                        print(f"Polling warning: {poll_exc}")
                        # Keep polling in case of transient DB lock
                        pass
            print(f"Result: {index_latency_str}\n")


# 6. Print Summary Table
print(f"{CLR_BLUE}======================================================================{CLR_RESET}")
print(f"{CLR_BLUE}                        BASELINE METRICS SUMMARY                      {CLR_RESET}")
print(f"{CLR_BLUE}======================================================================{CLR_RESET}")
print(f"| Metric Name                   | Value / Result                      |")
print(f"|-------------------------------|-------------------------------------|")
print(f"| Backend Health Latency        | {health_latency:<35} |")
print(f"| Frontend Build Size           | {dist_size_str[:35]:<35} |")

hist_dur = "N/A"
if latest_job_details and latest_job_details.get("started_at") and latest_job_details.get("completed_at"):
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        s_str = latest_job_details["started_at"].split("+")[0]
        c_str = latest_job_details["completed_at"].split("+")[0]
        t1 = datetime.strptime(s_str, fmt)
        t2 = datetime.strptime(c_str, fmt)
        hist_dur = f"{(t2 - t1).total_seconds():.1f} s ({latest_job_details['indexing_mode'].upper()})"
    except Exception:
        pass
print(f"| History Job Duration          | {hist_dur:<35} |")

if cmd_args.run_query:
    print(f"| Active Query Latency          | {query_latency_str:<35} |")
else:
    print(f"| Active Query Latency          | {'Not executed (skipped)':<35} |")

if cmd_args.run_index or cmd_args.run_incremental:
    # Get just the first line of the result string for table formatting
    idx_first_line = index_latency_str.split("\n")[0]
    print(f"| Active Indexing Duration      | {idx_first_line:<35} |")
else:
    print(f"| Active Indexing Duration      | {'Not executed (skipped)':<35} |")

print(f"{CLR_BLUE}======================================================================{CLR_RESET}")
