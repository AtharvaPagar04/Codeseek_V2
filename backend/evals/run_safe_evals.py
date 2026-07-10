#!/usr/bin/env python3
"""Safe evaluation runner and CI orchestrator for CodeSeek retrieval pipeline."""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def get_tail(text: str, max_lines: int = 15) -> str:
    """Extract the last max_lines of text."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSeek Safe Evaluation Runner")
    parser.add_argument("--session-id", required=True, help="Session ID from database.")
    parser.add_argument("--expected-repo-root", required=True, help="Expected repository root directory.")
    parser.add_argument("--expected-collection", required=True, help="Expected Qdrant collection name.")
    parser.add_argument("--output-dir", required=True, help="Directory to write all evaluation outputs.")
    
    # Optional flags
    parser.add_argument("--db-backend", default=os.getenv("CODESEEK_DB_BACKEND", "sqlite"), help="Database backend (default: sqlite).")
    parser.add_argument("--db-path", default=os.getenv("CODESEEK_DB_PATH", "/tmp/codeseek.sqlite3"), help="Path to sqlite database.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--timeout", type=int, default=1800, help="Subprocess timeout in seconds (default: 1800).")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed command execution output.")

    args = parser.parse_args()

    started_at = datetime.utcnow().isoformat() + "Z"
    start_time = time.time()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up known output files in the selected output-dir
    known_filenames = {
        "retrieval_latest.json",
        "conversation_latest.json",
        "eval_policy_summary.json",
        "eval_policy_summary.md",
        "safe_eval_summary.json",
        "safe_eval_summary.md",
    }
    for filename in known_filenames:
        target_file = output_dir / filename
        if target_file.is_file() and target_file.parent.resolve() == output_dir.resolve():
            try:
                target_file.unlink()
            except OSError:
                pass

    # 1. Define step structures
    step_definitions = []

    # Required: retrieval_eval
    step_definitions.append({
        "name": "retrieval_eval",
        "required": True,
        "command": [
            args.python_bin, "evals/retrieval_eval.py",
            "--session-id", args.session_id,
            "--golden", "evals/golden/golden_queries.yaml",
            "--output", str(output_dir / "retrieval_latest.json")
        ],
        "dependent_on": [],
        "output_path": str(output_dir / "retrieval_latest.json")
    })

    # Required: conversation_eval
    step_definitions.append({
        "name": "conversation_eval",
        "required": True,
        "command": [
            args.python_bin, "evals/conversation_eval.py",
            "--session-id", args.session_id,
            "--trees", "evals/golden/conversation_trees.yaml",
            "--output", str(output_dir / "conversation_latest.json")
        ],
        "dependent_on": [],
        "output_path": str(output_dir / "conversation_latest.json")
    })

    # Required: eval_policy_summary (Must construct command dynamically based on actual output existence)
    # We will finalize the command list in the execution loop itself.
    policy_step_def = {
        "name": "eval_policy_summary",
        "required": True,
        "command": [],  # Will build dynamically
        "dependent_on": ["retrieval_eval", "conversation_eval"],
        "output_path": str(output_dir / "eval_policy_summary.json")
    }
    step_definitions.append(policy_step_def)

    # 2. Execute loop
    executed_steps = []
    env = os.environ.copy()
    env["CODESEEK_DB_BACKEND"] = args.db_backend
    env["CODESEEK_DB_PATH"] = args.db_path
    env["PYTHONPATH"] = "."

    for step_def in step_definitions:
        name = step_def["name"]
        
        # Check dependencies
        deps_ok = True
        for dep in step_def["dependent_on"]:
            dep_step = next((s for s in executed_steps if s["name"] == dep), None)
            if not dep_step or dep_step["return_code"] != 0:
                deps_ok = False
                break
                
        if not deps_ok:
            print(f"Skipping step {name} due to upstream dependency failure.")
            ret_code = -1
            executed_steps.append({
                "name": name,
                "status": "ERROR",
                "command": step_def["command"],
                "return_code": ret_code,
                "duration_seconds": 0.0,
                "output_path": step_def["output_path"],
                "stdout_tail": "Skipped due to upstream dependency failure.",
                "stderr_tail": ""
            })
            if step_def.get("output_path"):
                out_p = Path(step_def["output_path"])
                if not out_p.exists() and out_p.parent.resolve() == output_dir.resolve():
                    try:
                        placeholder = {
                            "status": "ERROR",
                            "error": "step failed before producing report",
                            "step": name,
                            "return_code": ret_code
                        }
                        with open(out_p, "w", encoding="utf-8") as f:
                            json.dump(placeholder, f, indent=2)
                    except Exception:
                        pass
            continue

        # For policy summary, build the command dynamically to include only existing files
        if name == "eval_policy_summary":
            cmd = [
                args.python_bin, "evals/eval_policy_summary.py",
                "--retrieval-report", str(output_dir / "retrieval_latest.json"),
                "--conversation-report", str(output_dir / "conversation_latest.json"),
                "--output-json", str(output_dir / "eval_policy_summary.json"),
                "--output-md", str(output_dir / "eval_policy_summary.md")
            ]
            step_def["command"] = cmd

        print(f"Running step: {name}...")
        if args.verbose:
            print(f"Command: {' '.join(step_def['command'])}")

        step_start = time.time()
        try:
            res = subprocess.run(
                step_def["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=args.timeout
            )
            step_duration = round(time.time() - step_start, 2)
            ret_code = res.returncode
            stdout_str = res.stdout or ""
            stderr_str = res.stderr or ""
            status = "PASS" if ret_code == 0 else "ERROR"
        except subprocess.TimeoutExpired as te:
            step_duration = round(time.time() - step_start, 2)
            ret_code = -2
            stdout_str = te.stdout or ""
            stderr_str = (te.stderr or "") + f"\nTimeout expired after {args.timeout} seconds."
            status = "ERROR"
        except Exception as e:
            step_duration = round(time.time() - step_start, 2)
            ret_code = -3
            stdout_str = ""
            stderr_str = f"Execution exception: {str(e)}"
            status = "ERROR"

        executed_steps.append({
            "name": name,
            "status": status,
            "command": step_def["command"],
            "return_code": ret_code,
            "duration_seconds": step_duration,
            "output_path": step_def["output_path"],
            "stdout_tail": get_tail(stdout_str),
            "stderr_tail": get_tail(stderr_str)
        })

        if ret_code != 0 and step_def.get("output_path"):
            out_p = Path(step_def["output_path"])
            if not out_p.exists() and out_p.parent.resolve() == output_dir.resolve():
                try:
                    placeholder = {
                        "status": "ERROR",
                        "error": "step failed before producing report",
                        "step": name,
                        "return_code": ret_code
                    }
                    with open(out_p, "w", encoding="utf-8") as f:
                        json.dump(placeholder, f, indent=2)
                except Exception:
                    pass

    finished_at = datetime.utcnow().isoformat() + "Z"
    total_duration = round(time.time() - start_time, 2)

    # 3. Determine overall status and load policy summary metrics
    any_required_failed = False
    for step in executed_steps:
        # Match step name against required flag
        is_required = next((s["required"] for s in step_definitions if s["name"] == step["name"]), False)
        if is_required and step["return_code"] != 0:
            any_required_failed = True

    policy_summary_json = output_dir / "eval_policy_summary.json"
    policy_status = "ERROR"
    hard_gate_status = "ERROR"
    hard_gate_failures = []
    warnings = []
    diagnostics = []
    recommendation = "Safe evaluation run did not complete policy validation."

    # Check if eval_policy_summary step was run
    policy_step = next((s for s in executed_steps if s["name"] == "eval_policy_summary"), None)

    has_real_policy_report = False
    policy_data = None
    if policy_summary_json.exists():
        try:
            with open(policy_summary_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("error") != "step failed before producing report":
                has_real_policy_report = True
                policy_data = data
        except Exception:
            pass

    if has_real_policy_report:
        policy_status = policy_data.get("status", "ERROR")
        hard_gate_status = policy_data.get("hard_gate_status", "ERROR")
        hard_gate_failures = policy_data.get("hard_gate_failures", [])
        warnings = policy_data.get("warnings", [])
        diagnostics = policy_data.get("diagnostics", [])
        recommendation = policy_data.get("recommendation", "Policy check completed.")
    else:
        policy_status = "ERROR"
        hard_gate_status = "ERROR"
        if policy_step is not None and policy_step["return_code"] == -1:
            recommendation = "Upstream eval failed. Inspect step logs."
        else:
            recommendation = "Policy summary report missing or failed to run."
        
        # Collect failed step names (excluding eval_policy_summary itself if others failed)
        failed_steps = [s["name"] for s in executed_steps if s["return_code"] != 0 and s["name"] != "eval_policy_summary"]
        if not failed_steps:
            if policy_step is not None:
                failed_steps = ["eval_policy_summary"]
            else:
                failed_steps = ["eval_policy_summary (missing)"]
        hard_gate_failures = [f"Step failed: {step_name}" for step_name in failed_steps]

    # Compute overall status
    if any_required_failed or policy_status == "ERROR":
        overall_status = "ERROR"
    elif policy_status == "WARN":
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    # 4. Write final JSON summary
    summary_data = {
        "status": overall_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": total_duration,
        "session_id": args.session_id,
        "expected_repo_root": args.expected_repo_root,
        "expected_collection": args.expected_collection,
        "steps": executed_steps,
        "policy_summary_path": str(policy_summary_json),
        "hard_gate_status": hard_gate_status,
        "hard_gate_failures": hard_gate_failures,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "recommendation": recommendation
    }

    with open(output_dir / "safe_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Write final Markdown summary
    md_lines = [
        "# CodeSeek Safe Evaluation Run Summary",
        "",
        f"**Overall Status**: `{overall_status}`",
        f"**Started At**: {started_at}",
        f"**Finished At**: {finished_at}",
        f"**Duration**: {total_duration} seconds",
        f"**Session ID**: `{args.session_id}`",
        f"**Expected Repo Root**: `{args.expected_repo_root}`",
        f"**Expected Collection**: `{args.expected_collection}`",
        "",
        "## Execution Steps",
        "",
        "| Step Name | Status | Return Code | Duration (seconds) |",
        "|---|---|---|---|",
    ]
    for step in executed_steps:
        md_lines.append(f"| {step['name']} | `{step['status']}` | {step['return_code']} | {step['duration_seconds']} |")

    md_lines.extend([
        "",
        "## Gating Policy Summary",
        "",
        f"**Hard Gate Status**: `{hard_gate_status}`",
        "",
        "### Hard Gate Failures",
        ""
    ])
    if hard_gate_failures:
        for failure in hard_gate_failures:
            md_lines.append(f"- {failure}")
    else:
        md_lines.append("- None")

    md_lines.extend([
        "",
        "### Warnings",
        ""
    ])
    if warnings:
        for warning in warnings:
            md_lines.append(f"- {warning}")
    else:
        md_lines.append("- None")

    md_lines.extend([
        "",
        "### Diagnostics",
        ""
    ])
    if diagnostics:
        for diag in diagnostics:
            md_lines.append(f"- {diag}")
    else:
        md_lines.append("- None")

    md_lines.extend([
        "",
        "### Recommendation",
        "",
        recommendation,
        ""
    ])

    policy_md_file = output_dir / "eval_policy_summary.md"
    if policy_md_file.exists():
        try:
            policy_md_content = policy_md_file.read_text(encoding="utf-8")
            md_lines.extend([
                "---",
                "",
                "## Detailed Gating Policy Report",
                "",
                policy_md_content
            ])
        except Exception:
            pass

    with open(output_dir / "safe_eval_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Safe eval run finished. Overall status: {overall_status}")

if __name__ == "__main__":
    main()
