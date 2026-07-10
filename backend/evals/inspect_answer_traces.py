import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CodeSeek answer traces.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSONL trace file. Defaults to config ANSWER_TRACE_OUTPUT_PATH.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=5,
        help="Number of latest traces to print details for.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        try:
            from retrieval.config import ANSWER_TRACE_OUTPUT_PATH

            input_path = ANSWER_TRACE_OUTPUT_PATH
        except ImportError:
            input_path = str(
                Path(__file__).resolve().parent.parent
                / "evals"
                / "reports"
                / "answer_traces.jsonl"
            )

    path = Path(input_path)
    if not path.exists():
        print(f"Error: Trace file does not exist at '{path}'", file=sys.stderr)
        sys.exit(1)

    traces = []
    invalid_traces = 0

    required_fields = [
        "trace_id",
        "created_at",
        "schema_version",
        "question",
        "answer",
        "retrieved_contexts",
    ]

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
                traces.append(trace)

                # Validation
                missing = [field for field in required_fields if field not in trace]
                if missing:
                    invalid_traces += 1
                    continue

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                invalid_traces += 1

    total_traces = len(traces)
    print("========================================")
    print("         ANSWER TRACES SUMMARY")
    print("========================================")
    print(f"Total Traces Found:   {total_traces}")
    print(f"Invalid Traces:       {invalid_traces}")
    print("========================================")

    if not traces:
        return

    latest_n = min(args.latest, total_traces)
    print(f"\n--- Printing details for the latest {latest_n} traces ---")
    for i in range(total_traces - latest_n, total_traces):
        t = traces[i]
        print(f"\n[Trace {i + 1}] ID: {t.get('trace_id')}")
        print(f"  Created At:      {t.get('created_at')}")
        print(f"  Route:           {t.get('route')}")
        print(f"  Intent:          {t.get('reranker_intent')}")
        print(f"  Question:        {t.get('question')}")
        print(f"  Answer Snippet:  {str(t.get('answer'))[:100]}...")
        print(f"  Contexts Count:  {len(t.get('retrieved_contexts', []))}")
        print(f"  Latency:         {t.get('latency_ms')} ms")


if __name__ == "__main__":
    main()
