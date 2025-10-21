#!/usr/bin/env python3
"""
Compare the two security JSON outputs for Gemma results.

Usage:
  python compare_security_jsons.py [--results 3agent_gemma3n_e4b_results.csv] [--limit N]
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def pretty(obj):
    try:
        if isinstance(obj, str):
            # Attempt to parse JSON string first
            try:
                obj = json.loads(obj)
            except Exception:
                pass
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Compare masked vs general security JSONs for Gemma results")
    parser.add_argument("--results", default="3agent_gemma3n_e4b_results.csv", help="Path to Gemma results CSV")
    parser.add_argument("--limit", type=int, default=10, help="Print first N rows (set 0 to print all)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    total = len(df)
    n = total if args.limit in (None, 0) else min(args.limit, total)

    print(f"Comparing security JSONs in: {results_path} (showing {n}/{total})")
    print("=" * 80)

    for i, row in df.head(n).iterrows():
        msg_id = row.get("message_id", i)
        masked_raw = row.get("security_json", "") or ""
        general_raw = row.get("general_security_json", "") or ""

        # Parse JSON strings to extract fields like name/email/position for quick comparison
        def parse_json(s):
            try:
                return json.loads(s) if s else {}
            except Exception:
                return {}
        masked_obj = parse_json(masked_raw)
        general_obj = parse_json(general_raw)

        name_masked = masked_obj.get("name", "")
        name_general = general_obj.get("name", "")
        pos_masked = masked_obj.get("position", "")
        pos_general = general_obj.get("position", "")

        # Prepare pretty strings
        masked_pp = pretty(masked_raw)
        general_pp = pretty(general_raw)

        print(f"Row {i+1}  (message_id: {msg_id})")
        print("-" * 80)
        print(f"Name (masked: {name_masked}, general: {name_general})")
        print(f"Position (masked: {pos_masked}, general: {pos_general})")
        print()
        print("[masked security_json]")
        #print(masked_pp)
        print()
        print("[general_security_json]")
        #print(general_pp)
        print("=" * 80)


if __name__ == "__main__":
    main()


