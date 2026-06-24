import os
import sys
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from evals.run_safe_evals import main as run_safe_evals_main

@pytest.fixture
def temp_output_dir(tmp_path):
    d = tmp_path / "safe_eval_out"
    d.mkdir()
    return d

def test_command_building_and_safe_workflow(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-123",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "repository_chunks__local__codeseek",
        "--output-dir", str(temp_output_dir),
        "--verbose"
    ]

    captured_cmds = []

    def mock_run(cmd, *args, **kwargs):
        captured_cmds.append(cmd)
        
        # Write mock files as side effects of running
        if any("retrieval_eval.py" in arg for arg in cmd):
            ret_path = Path(cmd[cmd.index("--output") + 1])
            ret_path.write_text(json.dumps({"status": "PASS", "summary": {}}))
        elif any("conversation_eval.py" in arg for arg in cmd):
            conv_path = Path(cmd[cmd.index("--output") + 1])
            conv_path.write_text(json.dumps({"status": "PASS"}))
        elif any("eval_policy_summary.py" in arg for arg in cmd):
            out_json_path = Path(cmd[cmd.index("--output-json") + 1])
            out_json_path.write_text(json.dumps({
                "status": "PASS",
                "hard_gate_status": "PASS",
                "hard_gate_failures": [],
                "warnings": [],
                "diagnostics": [],
                "recommendation": "Passed policy tests."
            }))
            out_md_path = Path(cmd[cmd.index("--output-md") + 1])
            out_md_path.write_text("# Policy Report MD")

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Mock Success Out"
        mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    # 1. Builds retrieval command correctly
    ret_cmd = next((c for c in captured_cmds if any("retrieval_eval.py" in arg for arg in c)), None)
    assert ret_cmd is not None
    assert "--session-id" in ret_cmd
    assert "test-session-123" in ret_cmd
    assert "--golden" in ret_cmd

    # 2. Builds conversation command correctly
    conv_cmd = next((c for c in captured_cmds if any("conversation_eval.py" in arg for arg in c)), None)
    assert conv_cmd is not None
    assert "--session-id" in conv_cmd
    assert "test-session-123" in conv_cmd
    assert "--trees" in conv_cmd

    # 3. Runs required steps and writes summary JSON/Markdown
    summary_json = temp_output_dir / "safe_eval_summary.json"
    summary_md = temp_output_dir / "safe_eval_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    with open(summary_json, "r") as f:
        data = json.load(f)
    
    # 4. PASS when all subprocesses return 0 and policy summary is PASS
    assert data["status"] == "PASS"
    assert data["hard_gate_status"] == "PASS"
    
def test_status_warn_handling(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-warn",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        if any("retrieval_eval.py" in arg for arg in cmd):
            ret_path = Path(cmd[cmd.index("--output") + 1])
            ret_path.write_text(json.dumps({"status": "PASS", "summary": {}}))
        elif any("conversation_eval.py" in arg for arg in cmd):
            conv_path = Path(cmd[cmd.index("--output") + 1])
            conv_path.write_text(json.dumps({"status": "PASS"}))
        elif any("eval_policy_summary.py" in arg for arg in cmd):
            out_json_path = Path(cmd[cmd.index("--output-json") + 1])
            out_json_path.write_text(json.dumps({
                "status": "WARN",
                "hard_gate_status": "PASS",
                "hard_gate_failures": [],
                "warnings": ["Warning details"],
                "diagnostics": [],
                "recommendation": "Warnings present."
            }))

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success"
        mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "WARN"


def test_required_step_failure(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-fail",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        mock_process = MagicMock()
        if any("retrieval_eval.py" in arg for arg in cmd):
            mock_process.returncode = 1
            mock_process.stdout = ""
            mock_process.stderr = "Retrieval failed error details."
        else:
            mock_process.returncode = 0
            mock_process.stdout = "Success"
            mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "ERROR"
    # Dependent step eval_policy_summary should have failed/skipped
    policy_step = next(s for s in data["steps"] if s["name"] == "eval_policy_summary")
    assert policy_step["return_code"] == -1


def test_policy_summary_status_error(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-policy-error",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        if any("retrieval_eval.py" in arg for arg in cmd):
            ret_path = Path(cmd[cmd.index("--output") + 1])
            ret_path.write_text(json.dumps({"status": "PASS", "summary": {}}))
        elif any("conversation_eval.py" in arg for arg in cmd):
            conv_path = Path(cmd[cmd.index("--output") + 1])
            conv_path.write_text(json.dumps({"status": "PASS"}))
        elif any("eval_policy_summary.py" in arg for arg in cmd):
            out_json_path = Path(cmd[cmd.index("--output-json") + 1])
            out_json_path.write_text(json.dumps({
                "status": "ERROR",
                "hard_gate_status": "ERROR",
                "hard_gate_failures": ["Hard gate broken!"],
                "warnings": [],
                "diagnostics": [],
                "recommendation": "Error recommendation."
            }))

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success"
        mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "ERROR"
    assert data["hard_gate_status"] == "ERROR"


def test_timeout_marks_step_error(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-timeout",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        if any("retrieval_eval.py" in arg for arg in cmd):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1800, output="Partial std", stderr="Timeout expired")
        
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success"
        mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "ERROR"
    ret_step = next(s for s in data["steps"] if s["name"] == "retrieval_eval")
    assert ret_step["status"] == "ERROR"
    assert ret_step["return_code"] == -2


def test_missing_optional_reports_do_not_fail_runner(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-missing-opts",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        if any("retrieval_eval.py" in arg for arg in cmd):
            ret_path = Path(cmd[cmd.index("--output") + 1])
            ret_path.write_text(json.dumps({"status": "PASS", "summary": {}}))
        elif any("conversation_eval.py" in arg for arg in cmd):
            conv_path = Path(cmd[cmd.index("--output") + 1])
            conv_path.write_text(json.dumps({"status": "PASS"}))
        elif any("eval_policy_summary.py" in arg for arg in cmd):
            out_json_path = Path(cmd[cmd.index("--output-json") + 1])
            out_json_path.write_text(json.dumps({
                "status": "PASS",
                "hard_gate_status": "PASS",
                "hard_gate_failures": [],
                "warnings": [],
                "diagnostics": [],
                "recommendation": "Passed policy tests."
            }))

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success"
        mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASS"


def test_stale_reports_removed_on_failure(temp_output_dir):
    # Write stale report files
    stale_retrieval = temp_output_dir / "retrieval_latest.json"
    stale_retrieval.write_text(json.dumps({"status": "PASS", "message": "stale pass"}))
    
    stale_policy = temp_output_dir / "eval_policy_summary.json"
    stale_policy.write_text(json.dumps({"status": "PASS", "message": "stale pass"}))

    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-stale-cleanup",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 1  # Fail retrieval_eval immediately
        mock_process.stdout = ""
        mock_process.stderr = "Retrieval failed."
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    # retrieval_latest.json should contain the failure placeholder
    assert stale_retrieval.exists()
    with open(stale_retrieval, "r") as f:
        data = json.load(f)
    assert data["status"] == "ERROR"
    assert data["error"] == "step failed before producing report"

    # eval_policy_summary.json should contain the failure placeholder (since it skipped)
    assert stale_policy.exists()
    with open(stale_policy, "r") as f:
        data = json.load(f)
    assert data["status"] == "ERROR"
    assert data["error"] == "step failed before producing report"

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        summary = json.load(f)
    assert summary["status"] == "ERROR"
    assert summary["hard_gate_status"] == "ERROR"


def test_skipped_policy_summary_error_status(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "test-session-skipped-policy",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        mock_process = MagicMock()
        if any("retrieval_eval.py" in arg for arg in cmd):
            mock_process.returncode = 1
            mock_process.stdout = ""
            mock_process.stderr = "Retrieval failed."
        else:
            mock_process.returncode = 0
            mock_process.stdout = "Success"
            mock_process.stderr = ""
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        data = json.load(f)
    
    assert data["status"] == "ERROR"
    assert data["hard_gate_status"] == "ERROR"
    assert data["recommendation"] == "Upstream eval failed. Inspect step logs."
    assert "Step failed: retrieval_eval" in data["hard_gate_failures"]


def test_missing_session_failure_placeholder(temp_output_dir):
    argv = [
        "run_safe_evals.py",
        "--session-id", "missing-session-id",
        "--expected-repo-root", "/home/arch/DEV/CodeSeek",
        "--expected-collection", "coll",
        "--output-dir", str(temp_output_dir)
    ]

    def mock_run(cmd, *args, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Session not found in DB."
        return mock_process

    with patch("sys.argv", argv), patch("subprocess.run", side_effect=mock_run):
        run_safe_evals_main()

    ret_path = temp_output_dir / "retrieval_latest.json"
    conv_path = temp_output_dir / "conversation_latest.json"
    policy_path = temp_output_dir / "eval_policy_summary.json"

    for path in (ret_path, conv_path, policy_path):
        assert path.exists()
        with open(path, "r") as f:
            data = json.load(f)
        assert data["status"] == "ERROR"
        assert data["error"] == "step failed before producing report"

    summary_json = temp_output_dir / "safe_eval_summary.json"
    with open(summary_json, "r") as f:
        summary = json.load(f)
    assert summary["status"] == "ERROR"
    assert summary["hard_gate_status"] == "ERROR"
    assert "Step failed: retrieval_eval" in summary["hard_gate_failures"]
    assert "Step failed: conversation_eval" in summary["hard_gate_failures"]
