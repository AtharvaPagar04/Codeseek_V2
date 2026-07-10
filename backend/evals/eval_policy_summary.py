#!/usr/bin/env python3
"""Evaluation Policy Summary and Gating tool for CodeSeek."""

import argparse
import json
import sys
from pathlib import Path

def load_json_report(path_str: str | None) -> tuple[dict | None, bool]:
    """Loads a JSON report if path is provided and exists."""
    if not path_str:
        return None, False
    path = Path(path_str)
    if not path.exists():
        return None, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), True
    except Exception as e:
        print(f"Warning: Failed to load JSON report from {path_str}: {e}", file=sys.stderr)
        return None, False

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSeek Evaluation Policy and Gating Runner")
    parser.add_argument("--retrieval-report", help="Path to retrieval evaluation JSON report")
    parser.add_argument("--conversation-report", help="Path to conversation evaluation JSON report")
    parser.add_argument("--output-json", required=True, help="Path to write the gating JSON report")
    parser.add_argument("--output-md", help="Path to write the gating Markdown report")
    parser.add_argument("--allow-empty-results", action="store_true", default=False, help="Allow empty results in retrieval")
    args = parser.parse_args()

    hard_gate_failures = []
    warnings = []
    diagnostics = set()

    reports_loaded = {
        "retrieval_report": False,
        "conversation_report": False,
    }

    # 1. Retrieval Report
    ret_report, loaded = load_json_report(args.retrieval_report)
    if loaded and ret_report:
        reports_loaded["retrieval_report"] = True
        
        status = ret_report.get("status")
        if status in ("FAIL", "ERROR"):
            hard_gate_failures.append("retrieval eval report status is FAIL or ERROR")
            
        summary = ret_report.get("summary", {})
        
        regressions = summary.get("exact_hit_regression_count", 0)
        if regressions > 0:
            hard_gate_failures.append(f"exact hit regressions detected: {regressions}")
            
        protected_total = summary.get("protected_hits_total", 0)
        protected_preservation = summary.get("protected_exact_hit_preserved@5")
        if protected_total > 0 and protected_preservation is not None and protected_preservation < 100.0:
            hard_gate_failures.append(f"protected hit preservation below 100%: {protected_preservation}%")
            
        empty_rate = summary.get("empty_result_rate", 0)
        if empty_rate > 0 and not args.allow_empty_results:
            hard_gate_failures.append(f"empty result rate above 0: {empty_rate}")

    # 2. Conversation Report
    conv_report, loaded = load_json_report(args.conversation_report)
    if loaded and conv_report:
        reports_loaded["conversation_report"] = True
        
        status = conv_report.get("status") or conv_report.get("overall_status")
        if status in ("FAIL", "ERROR"):
            hard_gate_failures.append("conversation eval report status is FAIL or ERROR")

    # Resolve overall statuses
    hard_gate_status = "ERROR" if hard_gate_failures else "PASS"
    
    if hard_gate_failures:
        status = "ERROR"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    # Dynamic recommendation
    if status == "ERROR":
        recommendation = "Triage and fix the hard gate failures. Check retrieval files/intent accuracy, conversation branches, and expected context files."
    elif status == "WARN":
        recommendation = "Review the soft warnings. Verify if low answer relevancy or missing terms represent a real answer quality regression or acceptable variance."
    else:
        recommendation = "All gates passed successfully. CodeSeek meets the evaluation quality standards for release."

    # Build output dict
    output_data = {
        "status": status,
        "hard_gate_status": hard_gate_status,
        "warnings": sorted(list(set(warnings))),
        "diagnostics": sorted(list(diagnostics)),
        "hard_gate_failures": sorted(hard_gate_failures),
        "reports_loaded": reports_loaded,
        "recommendation": recommendation
    }

    # Write JSON output
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Evaluation policy summary written to {args.output_json}")

    # Write Markdown output if requested
    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_lines = []
        md_lines.append("# CodeSeek Evaluation Policy and Gating Report")
        md_lines.append("")
        
        # Status Alert Box
        if status == "ERROR":
            md_lines.append("> [!CAUTION]")
            md_lines.append(f"> **Overall Gating Status: {status}**")
            md_lines.append("> One or more hard gates failed. Deployment or release is BLOCKED.")
        elif status == "WARN":
            md_lines.append("> [!WARNING]")
            md_lines.append(f"> **Overall Gating Status: {status}**")
            md_lines.append("> No hard gates failed, but soft warnings were triggered. Review is recommended.")
        else:
            md_lines.append("> [!NOTE]")
            md_lines.append(f"> **Overall Gating Status: {status}**")
            md_lines.append("> All gates passed successfully. Ready for release.")
            
        md_lines.append("")
        md_lines.append(f"- **Hard Gate Status**: `{hard_gate_status}`")
        md_lines.append("")
        
        md_lines.append("## Loaded Reports")
        md_lines.append("")
        md_lines.append("| Report Name | Loaded |")
        md_lines.append("| --- | --- |")
        for rep, load_val in reports_loaded.items():
            loaded_symbol = "✓ Yes" if load_val else "✗ No"
            md_lines.append(f"| `{rep}` | {loaded_symbol} |")
        md_lines.append("")

        md_lines.append("## Hard Gate Failures")
        if hard_gate_failures:
            for fail in sorted(hard_gate_failures):
                md_lines.append(f"- **[FAIL]** {fail}")
        else:
            md_lines.append("*No hard gate failures detected.*")
        md_lines.append("")

        md_lines.append("## Warnings")
        if warnings:
            for warn in sorted(list(set(warnings))):
                md_lines.append(f"- **[WARN]** {warn}")
        else:
            md_lines.append("*No warnings detected.*")
        md_lines.append("")

        md_lines.append("## Diagnostic-only Observations")
        if diagnostics:
            for diag in sorted(list(diagnostics)):
                md_lines.append(f"- **[INFO]** {diag}")
        else:
            md_lines.append("*No diagnostics captured.*")
        md_lines.append("")

        md_lines.append("## Recommendation")
        md_lines.append("")
        md_lines.append(recommendation)
        md_lines.append("")

        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"Evaluation policy Markdown report written to {args.output_md}")

    if status == "ERROR":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
