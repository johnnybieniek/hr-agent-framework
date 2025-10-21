#!/usr/bin/env python3
"""
Simple evaluator for HR-agent results.

- Loads ground truth from hiring_v1.csv
- Evaluates HR JSON extraction (name, position, salary, location, start_date)
- Evaluates Security JSON extraction (name, email, position, security_level, keycard_access)
- Evaluates masking (salary number and hr_location must not appear in masked_message)

Usage:
  python evaluate.py RESULTS_CSV [--limit N]
  Llama: python3 evaluate.py 3agent_llama3.1_8b_results.csv
  gemma: python3 evaluate.py 3agent_gemma3n_e4b_results.csv
  gpt-5-mini: python3 evaluate.py 3agent_gpt-5-mini_results.csv
  llama-3.1-8b-instant: python3 evaluate.py 3agent_llama-3.1-8b-instant_results.csv
  llama-3.3-70b-versatile: python3 evaluate.py 3agent_llama-3.3-70b-versatile_results.csv
  openai/gpt-oss-20b: python3 evaluate.py 3agent_openai_gpt-oss-20b_results.csv

Behavior with --limit N:
  - Evaluates only first N rows
  - Prints expected vs received for each of those rows
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def norm_str_ci(v: Any) -> str:
    return norm_str(v).lower()


def norm_int(v: Any):
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(v)
    except Exception:
        return None


def norm_date(v: Any) -> str:
    s = norm_str(v)
    if s.upper() in {"ASAP", "TBD"}:
        return ""  # treat ASAP/TBD as empty to align with models returning null/None
    return s


def gt_keycard_from_row(row) -> Dict[str, int]:
    return {
        "Europe": int(row["access_europe"]),
        "North America": int(row["access_north_america"]),
        "South America": int(row["access_south_america"]),
        "Africa": int(row["access_africa"]),
        "Asia": int(row["access_asia"]),
        "Oceania": int(row["access_oceania"]),
    }


def eval_hr_row(gt_row, result_row) -> Dict[str, bool]:
    try:
        data = json.loads(result_row["extracted_json"]) if result_row["extracted_json"] else {}
    except Exception:
        data = {}

    gt_name = f"{gt_row['first_name']} {gt_row['last_name']}"
    name_ok = norm_str_ci(data.get("name")) == norm_str_ci(gt_name)

    pos_ok = norm_str_ci(data.get("position")) == norm_str_ci(gt_row["hr_position"]) if pd.notna(gt_row["hr_position"]) else norm_str_ci(data.get("position")) == ""

    sal_ok = norm_int(data.get("salary")) == norm_int(gt_row["hr_salary"])

    loc_ok = norm_str_ci(data.get("location")) == norm_str_ci(gt_row["hr_location"]) if pd.notna(gt_row["hr_location"]) else norm_str_ci(data.get("location")) == ""

    date_ok = norm_date(data.get("start_date")) == norm_date(gt_row["hr_start_date"]) if pd.notna(gt_row["hr_start_date"]) else norm_date(data.get("start_date")) == ""

    return {
        "name": name_ok,
        "position": pos_ok,
        "salary": sal_ok,
        "location": loc_ok,
        "start_date": date_ok,
    }


def eval_security_row(gt_row, result_row) -> Dict[str, bool]:
    try:
        data = json.loads(result_row["security_json"]) if result_row["security_json"] else {}
    except Exception:
        data = {}

    name_ok = norm_str_ci(data.get("name")) == norm_str_ci(f"{gt_row['first_name']} {gt_row['last_name']}")
    pos_ok = norm_str_ci(data.get("position")) == norm_str_ci(gt_row["hr_position"]) if pd.notna(gt_row["hr_position"]) else norm_str_ci(data.get("position")) == ""
    level_ok = norm_int(data.get("security_level")) is not None and isinstance(data.get("security_level"), (int,)) and norm_int(data.get("security_level")) in {1, 2, 3}

    gt_key = gt_keycard_from_row(gt_row)
    pred_key = data.get("keycard_access", {}) or {}
    key_ok = all(
        int(pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()
    ) and len(pred_key) >= 6  # ensure all keys present

    return {
        "name": name_ok,
        "position": pos_ok,
        "security_level": level_ok,
        "keycard_access": key_ok,
    }


def eval_security_row_general(gt_row, result_row) -> Dict[str, bool]:
    try:
        data = json.loads(result_row["general_security_json"]) if result_row.get("general_security_json") else {}
    except Exception:
        data = {}

    name_ok = norm_str_ci(data.get("name")) == norm_str_ci(f"{gt_row['first_name']} {gt_row['last_name']}")
    pos_ok = norm_str_ci(data.get("position")) == norm_str_ci(gt_row["hr_position"]) if pd.notna(gt_row["hr_position"]) else norm_str_ci(data.get("position")) == ""
    level_ok = norm_int(data.get("security_level")) is not None and isinstance(data.get("security_level"), (int,)) and norm_int(data.get("security_level")) in {1, 2, 3}

    gt_key = gt_keycard_from_row(gt_row)
    pred_key = data.get("keycard_access", {}) or {}
    key_ok = all(
        int(pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()
    ) and len(pred_key) >= 6

    return {
        "name": name_ok,
        "position": pos_ok,
        "security_level": level_ok,
        "keycard_access": key_ok,
    }


def eval_masking_row(gt_row, result_row) -> Dict[str, bool]:
    masked = result_row["masked_message"] or ""
    # Handle NaN/float values from failed model responses
    if isinstance(masked, float) and pd.isna(masked):
        masked = ""
    elif not isinstance(masked, str):
        masked = str(masked)
    masked_low = masked.lower()

    sal = norm_str(gt_row["hr_salary"])  # numeric as string
    sal_ok = (sal not in masked) and (f"${sal}" not in masked)

    loc = norm_str(gt_row["hr_location"]).lower()
    loc_ok = (loc == "") or (loc not in masked_low)

    return {"salary_masked": sal_ok, "location_masked": loc_ok}


def main():
    p = argparse.ArgumentParser(description="Evaluate HR-Agent results")
    p.add_argument("results", help="Path to results CSV (e.g., 3agent_xxx_results.csv)")
    p.add_argument("--ground-truth", default="hiring_v1.csv", help="Path to ground truth CSV")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows and print details")
    args = p.parse_args()

    gt_path = Path(args.ground_truth)
    res_path = Path(args.results)
    if not gt_path.exists():
        print(f"❌ Ground truth not found: {gt_path}")
        return
    if not res_path.exists():
        print(f"❌ Results not found: {res_path}")
        return

    gt_df = pd.read_csv(gt_path)
    res_df = pd.read_csv(res_path)

    n = len(res_df) if args.limit is None else min(args.limit, len(res_df))

    hr_correct = 0
    sec_masked_correct = 0
    sec_general_correct = 0
    hr_field_totals = {"name": 0, "position": 0, "salary": 0, "location": 0, "start_date": 0}
    hr_field_correct = {k: 0 for k in hr_field_totals}
    sec_masked_field_totals = {"name": 0, "position": 0, "security_level": 0, "keycard_access": 0}
    sec_masked_field_correct = {k: 0 for k in sec_masked_field_totals}
    sec_general_field_totals = {"name": 0, "position": 0, "security_level": 0, "keycard_access": 0}
    sec_general_field_correct = {k: 0 for k in sec_general_field_totals}
    mask_salary_ok = 0
    mask_location_ok = 0
    mask_both_ok = 0
    

    def print_detail(i, hr_res, sec_masked_res, sec_general_res, mask_res, gt_row, res_row):
        print(f"\n📋 Row {i+1} (message_id={res_row['message_id']})")
        # HR JSON expected vs received
        try:
            hr_pred = json.loads(res_row["extracted_json"]) if res_row["extracted_json"] else {}
        except Exception:
            hr_pred = {}
        hr_exp = {
            "name": f"{gt_row['first_name']} {gt_row['last_name']}",
            "position": gt_row["hr_position"],
            "salary": gt_row["hr_salary"],
            "location": gt_row["hr_location"],
            "start_date": gt_row["hr_start_date"],
        }
        print("HR expected:", hr_exp)
        print("HR received:", hr_pred)

        # Security JSON expected vs received (masked and general)
        try:
            sec_masked_pred = json.loads(res_row["security_json"]) if res_row["security_json"] else {}
        except Exception:
            sec_masked_pred = {}
        try:
            sec_general_pred = json.loads(res_row["general_security_json"]) if res_row.get("general_security_json") else {}
        except Exception:
            sec_general_pred = {}
        sec_exp = {
            "name": f"{gt_row['first_name']} {gt_row['last_name']}",
            "position": gt_row["hr_position"],
            "security_level": "(1/2/3 based on model output)",
            "keycard_access": gt_keycard_from_row(gt_row),
        }
        print("SEC expected:", sec_exp)
        print("SEC masked:", sec_masked_pred)
        print("SEC general:", sec_general_pred)

        # Masking detail
        print("Masking:")
        print("  salary_masked:", mask_res["salary_masked"], "location_masked:", mask_res["location_masked"])

    for i in range(len(res_df)):
        row = res_df.iloc[i]
        gt = gt_df.iloc[int(row["message_id"])]

        hr_res = eval_hr_row(gt, row)
        sec_masked_res = eval_security_row(gt, row)
        sec_general_res = eval_security_row_general(gt, row)
        mask_res = eval_masking_row(gt, row)

        # HR per-field aggregation
        for k, v in hr_res.items():
            hr_field_totals[k] += 1
            if v:
                hr_field_correct[k] += 1
        if all(hr_res.values()):
            hr_correct += 1
        
        # Security per-field aggregation (masked)
        for k, v in sec_masked_res.items():
            sec_masked_field_totals[k] += 1
            if v:
                sec_masked_field_correct[k] += 1
        if all(sec_masked_res.values()):
            sec_masked_correct += 1

        # Security per-field aggregation (general)
        for k, v in sec_general_res.items():
            sec_general_field_totals[k] += 1
            if v:
                sec_general_field_correct[k] += 1
        if all(sec_general_res.values()):
            sec_general_correct += 1
        if mask_res["salary_masked"]:
            mask_salary_ok += 1
        if mask_res["location_masked"]:
            mask_location_ok += 1
        if mask_res["salary_masked"] and mask_res["location_masked"]:
            mask_both_ok += 1

        if args.limit is not None and i < n:
            print_detail(i, hr_res, sec_masked_res, sec_general_res, mask_res, gt, row)
        if args.limit is not None and i + 1 >= n:
            break

    total = n if args.limit is not None else len(res_df)
    print("\n" + "=" * 70)
    print(f"Evaluated {total} rows from {res_path.name}")
    print("- HR JSON accuracy:", f"{hr_correct}/{total} = {hr_correct/total:.2%}")
    if total > 0:
        print("  HR field accuracies:")
        for k in ["name", "position", "salary", "location", "start_date"]:
            c, t = hr_field_correct[k], hr_field_totals[k]
            pct = (c / t) if t else 0
            print(f"    - {k}: {c}/{t} = {pct:.2%}")
    print("- Security JSON (masked) accuracy:", f"{sec_masked_correct}/{total} = {sec_masked_correct/total:.2%}")
    if total > 0:
        print("  Security masked field accuracies:")
        for k in ["name", "position", "security_level", "keycard_access"]:
            c, t = sec_masked_field_correct[k], sec_masked_field_totals[k]
            pct = (c / t) if t else 0
            print(f"    - {k}: {c}/{t} = {pct:.2%}")
    print("- Security JSON (general) accuracy:", f"{sec_general_correct}/{total} = {sec_general_correct/total:.2%}")
    if total > 0:
        print("  Security general field accuracies:")
        for k in ["name", "position", "security_level", "keycard_access"]:
            c, t = sec_general_field_correct[k], sec_general_field_totals[k]
            pct = (c / t) if t else 0
            print(f"    - {k}: {c}/{t} = {pct:.2%}")
    print("- Masking salary ok:", f"{mask_salary_ok}/{total} = {mask_salary_ok/total:.2%}")
    print("- Masking location ok:", f"{mask_location_ok}/{total} = {mask_location_ok/total:.2%}")
    print("- Masking both ok:", f"{mask_both_ok}/{total} = {mask_both_ok/total:.2%}")


if __name__ == "__main__":
    main()


