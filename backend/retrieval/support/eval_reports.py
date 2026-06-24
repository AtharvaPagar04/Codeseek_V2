import os
import json
from datetime import datetime
from pathlib import Path

def get_latest_evaluation_report(session_id: str | None = None) -> dict:
    """Resolve and load the latest safe evaluation report."""
    backend_root = Path(__file__).resolve().parent.parent
    report_path = backend_root.parent / "evals" / "reports" / "safe_eval_latest" / "safe_eval_summary.json"
    report_path = report_path.resolve()

    if not report_path.exists():
        return {
            "available": False,
            "status": "UNKNOWN",
            "message": "No safe evaluation report found. Run evals/run_safe_evals.py first.",
            "steps": [],
            "hard_gate_failures": [],
            "warnings": [],
            "diagnostics": []
        }

    try:
        content = report_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "status": "ERROR",
            "message": f"Invalid JSON in safe evaluation report: {str(exc)}",
            "steps": [],
            "hard_gate_failures": [],
            "warnings": [],
            "diagnostics": []
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "ERROR",
            "message": f"Error reading safe evaluation report: {str(exc)}",
            "steps": [],
            "hard_gate_failures": [],
            "warnings": [],
            "diagnostics": []
        }

    result = {
        "available": True,
        "report_path": str(report_path),
        "loaded_at": datetime.utcnow().isoformat() + "Z"
    }
    result.update(data)
    return result
