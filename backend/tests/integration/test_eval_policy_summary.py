import json
import subprocess
import sys
from pathlib import Path
import pytest

@pytest.fixture
def tmp_reports_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d

def run_policy_summary(args_list):
    script_path = str(Path(__file__).resolve().parents[2] / "evals" / "eval_policy_summary.py")
    cmd = [sys.executable, script_path] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_pass_clean_reports(tmp_reports_dir):
    # 1. PASS when core reports pass.
    ret_path = tmp_reports_dir / "retrieval.json"
    conv_path = tmp_reports_dir / "conversation.json"
    out_json = tmp_reports_dir / "output.json"
    out_md = tmp_reports_dir / "output.md"

    # Write clean reports
    ret_path.write_text(json.dumps({
        "status": "PASS",
        "summary": {
            "exact_hit_regression_count": 0,
            "protected_hits_total": 5,
            "protected_exact_hit_preserved@5": 100.0,
            "empty_result_rate": 0.0
        }
    }))
    conv_path.write_text(json.dumps({
        "status": "PASS"
    }))

    args = [
        "--retrieval-report", str(ret_path),
        "--conversation-report", str(conv_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md)
    ]

    res = run_policy_summary(args)
    assert res.returncode == 0

    # Read output
    with open(out_json, "r") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["hard_gate_status"] == "PASS"
    assert len(data["warnings"]) == 0
    assert len(data["hard_gate_failures"]) == 0
    assert len(data["diagnostics"]) == 0

def test_error_retrieval_fail(tmp_reports_dir):
    # 2. ERROR when retrieval status is FAIL.
    ret_path = tmp_reports_dir / "retrieval.json"
    out_json = tmp_reports_dir / "output.json"

    ret_path.write_text(json.dumps({
        "status": "FAIL",
        "summary": {}
    }))

    res = run_policy_summary([
        "--retrieval-report", str(ret_path),
        "--output-json", str(out_json)
    ])
    assert res.returncode == 1

    with open(out_json, "r") as f:
        data = json.load(f)

    assert data["status"] == "ERROR"
    assert data["hard_gate_status"] == "ERROR"
    assert "retrieval eval report status is FAIL or ERROR" in data["hard_gate_failures"]

def test_error_conversation_error(tmp_reports_dir):
    # 3. ERROR when conversation status is ERROR.
    conv_path = tmp_reports_dir / "conversation.json"
    out_json = tmp_reports_dir / "output.json"

    conv_path.write_text(json.dumps({
        "status": "ERROR"
    }))

    res = run_policy_summary([
        "--conversation-report", str(conv_path),
        "--output-json", str(out_json)
    ])
    assert res.returncode == 1

    with open(out_json, "r") as f:
        data = json.load(f)

    assert data["status"] == "ERROR"
    assert data["hard_gate_status"] == "ERROR"
    assert "conversation eval report status is FAIL or ERROR" in data["hard_gate_failures"]

def test_json_and_markdown_written(tmp_reports_dir):
    # 4. JSON and Markdown outputs are written.
    ret_path = tmp_reports_dir / "retrieval.json"
    out_json = tmp_reports_dir / "output.json"
    out_md = tmp_reports_dir / "output.md"

    ret_path.write_text(json.dumps({
        "status": "PASS"
    }))

    res = run_policy_summary([
        "--retrieval-report", str(ret_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md)
    ])
    assert res.returncode == 0

    assert out_json.exists()
    assert out_md.exists()
    
    md_content = out_md.read_text()
    assert "# CodeSeek Evaluation Policy and Gating Report" in md_content

def test_missing_optional_reports_gracefully(tmp_reports_dir):
    # 5. Missing optional reports are handled gracefully.
    out_json = tmp_reports_dir / "output.json"

    res = run_policy_summary([
        "--retrieval-report", "nonexistent_retrieval.json",
        "--conversation-report", "nonexistent_conversation.json",
        "--output-json", str(out_json)
    ])
    assert res.returncode == 0

    with open(out_json, "r") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["reports_loaded"]["retrieval_report"] is False
    assert data["reports_loaded"]["conversation_report"] is False
