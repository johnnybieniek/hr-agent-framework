#!/usr/bin/env python3
"""
Detailed evaluator for HR-agent results that prints expected vs received for each field.

- Loads ground truth from hiring_v1.csv
- Evaluates HR JSON extraction (name, position, salary, location, start_date)
- Evaluates Security JSON extraction (name, position, security_level, keycard_access)
- Evaluates masking (salary number and hr_location must not appear in masked_message)
- Prints detailed expected vs received for each field with correct/wrong labels

Usage:
  python print_evaluate.py RESULTS_CSV [--limit N]

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
    masked_low = masked.lower()

    sal = norm_str(gt_row["hr_salary"])  # numeric as string
    sal_ok = (sal not in masked) and (f"${sal}" not in masked)

    loc = norm_str(gt_row["hr_location"]).lower()
    loc_ok = (loc == "") or (loc not in masked_low)

    return {"salary_masked": sal_ok, "location_masked": loc_ok}


def has_any_errors(gt_row, res_row):
    """Check if there are any errors in this record."""
    # HR JSON evaluation
    try:
        hr_data = json.loads(res_row["extracted_json"]) if res_row["extracted_json"] else {}
    except Exception:
        hr_data = {}
    
    gt_name = f"{gt_row['first_name']} {gt_row['last_name']}"
    hr_name = hr_data.get("name", "")
    name_ok = norm_str_ci(hr_name) == norm_str_ci(gt_name)
    
    gt_pos = gt_row["hr_position"] if pd.notna(gt_row["hr_position"]) else ""
    hr_pos = hr_data.get("position", "")
    pos_ok = norm_str_ci(hr_pos) == norm_str_ci(gt_pos)
    
    gt_sal = gt_row["hr_salary"]
    hr_sal = hr_data.get("salary")
    sal_ok = norm_int(hr_sal) == norm_int(gt_sal)
    
    gt_loc = gt_row["hr_location"] if pd.notna(gt_row["hr_location"]) else ""
    hr_loc = hr_data.get("location", "")
    loc_ok = norm_str_ci(hr_loc) == norm_str_ci(gt_loc)
    
    gt_date = gt_row["hr_start_date"] if pd.notna(gt_row["hr_start_date"]) else ""
    hr_date = hr_data.get("start_date", "")
    date_ok = norm_date(hr_date) == norm_date(gt_date)
    
    # Security JSON evaluation (masked)
    try:
        sec_masked_data = json.loads(res_row["security_json"]) if res_row["security_json"] else {}
    except Exception:
        sec_masked_data = {}
    
    sec_name_ok = norm_str_ci(sec_masked_data.get("name", "")) == norm_str_ci(gt_name)
    sec_pos_ok = norm_str_ci(sec_masked_data.get("position", "")) == norm_str_ci(gt_pos)
    sec_level = sec_masked_data.get("security_level")
    sec_level_ok = norm_int(sec_level) is not None and isinstance(sec_level, (int,)) and norm_int(sec_level) in {1, 2, 3}
    
    gt_key = gt_keycard_from_row(gt_row)
    pred_key = sec_masked_data.get("keycard_access", {}) or {}
    sec_key_ok = all(int(pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(pred_key) >= 6
    
    # Security JSON evaluation (general)
    try:
        sec_general_data = json.loads(res_row["general_security_json"]) if res_row.get("general_security_json") else {}
    except Exception:
        sec_general_data = {}
    
    sec_gen_name_ok = norm_str_ci(sec_general_data.get("name", "")) == norm_str_ci(gt_name)
    sec_gen_pos_ok = norm_str_ci(sec_general_data.get("position", "")) == norm_str_ci(gt_pos)
    sec_gen_level = sec_general_data.get("security_level")
    sec_gen_level_ok = norm_int(sec_gen_level) is not None and isinstance(sec_gen_level, (int,)) and norm_int(sec_gen_level) in {1, 2, 3}
    
    sec_gen_pred_key = sec_general_data.get("keycard_access", {}) or {}
    sec_gen_key_ok = all(int(sec_gen_pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(sec_gen_pred_key) >= 6
    
    # Masking evaluation
    masked = res_row["masked_message"] or ""
    sal = norm_str(gt_row["hr_salary"])
    sal_ok = (sal not in masked) and (f"${sal}" not in masked)
    
    loc = norm_str(gt_row["hr_location"]).lower()
    loc_ok = (loc == "") or (loc not in masked.lower())
    
    # Return True if ANY field has an error
    return not all([
        name_ok, pos_ok, sal_ok, loc_ok, date_ok,  # HR fields
        sec_name_ok, sec_pos_ok, sec_level_ok, sec_key_ok,  # Security masked
        sec_gen_name_ok, sec_gen_pos_ok, sec_gen_level_ok, sec_gen_key_ok,  # Security general
        sal_ok, loc_ok  # Masking
    ])


def print_detailed_evaluation(i, gt_row, res_row):
    print(f"\n{'='*80}")
    print(f"📋 ROW {i+1} (message_id={res_row['message_id']})")
    print(f"{'='*80}")
    
    # HR JSON evaluation
    print("\n🔍 HR JSON EVALUATION:")
    print("-" * 40)
    
    try:
        hr_data = json.loads(res_row["extracted_json"]) if res_row["extracted_json"] else {}
    except Exception:
        hr_data = {}
    
    # Name
    gt_name = f"{gt_row['first_name']} {gt_row['last_name']}"
    hr_name = hr_data.get("name", "")
    name_ok = norm_str_ci(hr_name) == norm_str_ci(gt_name)
    print(f"Name:     Expected='{gt_name}' | Received='{hr_name}' | {'✅ CORRECT' if name_ok else '❌ WRONG'}")
    
    # Position
    gt_pos = gt_row["hr_position"] if pd.notna(gt_row["hr_position"]) else ""
    hr_pos = hr_data.get("position", "")
    pos_ok = norm_str_ci(hr_pos) == norm_str_ci(gt_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{hr_pos}' | {'✅ CORRECT' if pos_ok else '❌ WRONG'}")
    
    # Salary
    gt_sal = gt_row["hr_salary"]
    hr_sal = hr_data.get("salary")
    sal_ok = norm_int(hr_sal) == norm_int(gt_sal)
    print(f"Salary:   Expected='{gt_sal}' | Received='{hr_sal}' | {'✅ CORRECT' if sal_ok else '❌ WRONG'}")
    
    # Location
    gt_loc = gt_row["hr_location"] if pd.notna(gt_row["hr_location"]) else ""
    hr_loc = hr_data.get("location", "")
    loc_ok = norm_str_ci(hr_loc) == norm_str_ci(gt_loc)
    print(f"Location: Expected='{gt_loc}' | Received='{hr_loc}' | {'✅ CORRECT' if loc_ok else '❌ WRONG'}")
    
    # Start Date
    gt_date = gt_row["hr_start_date"] if pd.notna(gt_row["hr_start_date"]) else ""
    hr_date = hr_data.get("start_date", "")
    date_ok = norm_date(hr_date) == norm_date(gt_date)
    print(f"Start:    Expected='{gt_date}' | Received='{hr_date}' | {'✅ CORRECT' if date_ok else '❌ WRONG'}")
    
    # Security JSON evaluation (masked)
    print("\n🔒 SECURITY JSON EVALUATION (MASKED):")
    print("-" * 40)
    
    try:
        sec_masked_data = json.loads(res_row["security_json"]) if res_row["security_json"] else {}
    except Exception:
        sec_masked_data = {}
    
    # Name
    sec_name = sec_masked_data.get("name", "")
    sec_name_ok = norm_str_ci(sec_name) == norm_str_ci(gt_name)
    print(f"Name:     Expected='{gt_name}' | Received='{sec_name}' | {'✅ CORRECT' if sec_name_ok else '❌ WRONG'}")
    
    # Position
    sec_pos = sec_masked_data.get("position", "")
    sec_pos_ok = norm_str_ci(sec_pos) == norm_str_ci(gt_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{sec_pos}' | {'✅ CORRECT' if sec_pos_ok else '❌ WRONG'}")
    
    # Security Level
    sec_level = sec_masked_data.get("security_level")
    sec_level_ok = norm_int(sec_level) is not None and isinstance(sec_level, (int,)) and norm_int(sec_level) in {1, 2, 3}
    print(f"Level:    Expected='1/2/3' | Received='{sec_level}' | {'✅ CORRECT' if sec_level_ok else '❌ WRONG'}")
    
    # Keycard Access
    gt_key = gt_keycard_from_row(gt_row)
    pred_key = sec_masked_data.get("keycard_access", {}) or {}
    key_ok = all(int(pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(pred_key) >= 6
    print(f"Keycard:  Expected='{gt_key}' | Received='{pred_key}' | {'✅ CORRECT' if key_ok else '❌ WRONG'}")
    
    # Security JSON evaluation (general)
    print("\n🔒 SECURITY JSON EVALUATION (GENERAL):")
    print("-" * 40)
    
    try:
        sec_general_data = json.loads(res_row["general_security_json"]) if res_row.get("general_security_json") else {}
    except Exception:
        sec_general_data = {}
    
    # Name
    sec_gen_name = sec_general_data.get("name", "")
    sec_gen_name_ok = norm_str_ci(sec_gen_name) == norm_str_ci(gt_name)
    print(f"Name:     Expected='{gt_name}' | Received='{sec_gen_name}' | {'✅ CORRECT' if sec_gen_name_ok else '❌ WRONG'}")
    
    # Position
    sec_gen_pos = sec_general_data.get("position", "")
    sec_gen_pos_ok = norm_str_ci(sec_gen_pos) == norm_str_ci(gt_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{sec_gen_pos}' | {'✅ CORRECT' if sec_gen_pos_ok else '❌ WRONG'}")
    
    # Security Level
    sec_gen_level = sec_general_data.get("security_level")
    sec_gen_level_ok = norm_int(sec_gen_level) is not None and isinstance(sec_gen_level, (int,)) and norm_int(sec_gen_level) in {1, 2, 3}
    print(f"Level:    Expected='1/2/3' | Received='{sec_gen_level}' | {'✅ CORRECT' if sec_gen_level_ok else '❌ WRONG'}")
    
    # Keycard Access
    sec_gen_pred_key = sec_general_data.get("keycard_access", {}) or {}
    sec_gen_key_ok = all(int(sec_gen_pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(sec_gen_pred_key) >= 6
    print(f"Keycard:  Expected='{gt_key}' | Received='{sec_gen_pred_key}' | {'✅ CORRECT' if sec_gen_key_ok else '❌ WRONG'}")
    
    # Masking evaluation
    print("\n🎭 MASKING EVALUATION:")
    print("-" * 40)
    
    masked = res_row["masked_message"] or ""
    sal = norm_str(gt_row["hr_salary"])
    print("Salary: ",sal)
    sal_ok = (sal not in masked) and (f"${sal}" not in masked)
    print(f"Salary:   Expected='NOT in masked' | Found='{sal in masked}' | {'✅ CORRECT' if sal_ok else '❌ WRONG'}")
    
    loc = norm_str(gt_row["hr_location"]).lower()
    print("Location: ",loc)
    loc_ok = (loc == "") or (loc not in masked.lower())
    print(f"Location: Expected='NOT in masked' | Found='{loc in masked.lower()}' | {'✅ CORRECT' if loc_ok else '❌ WRONG'}")

    print("Masked message: ", masked)


def main():
    p = argparse.ArgumentParser(description="Detailed evaluation of HR-Agent results")
    p.add_argument("results", help="Path to results CSV (e.g., 3agent_xxx_results.csv)")
    p.add_argument("--ground-truth", default="hiring_v1.csv", help="Path to ground truth CSV")
    p.add_argument("--limit", type=int, default=100, help="Evaluate only first N rows (default: 100)")
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

    n = min(args.limit, len(res_df))
    print(f"🔍 DETAILED EVALUATION: {n} rows from {res_path.name}")
    print(f"Ground truth: {gt_path.name}")

    error_count = 0
    total_processed = 0
    
    for i in range(n):
        row = res_df.iloc[i]
        gt = gt_df.iloc[int(row["message_id"])]
        total_processed += 1
        
        # Only print if there are errors
        if has_any_errors(gt, row):
            error_count += 1
            print_detailed_evaluation(i, gt, row)
        else:
            print(f"✅ ROW {i+1} (message_id={row['message_id']}) - ALL CORRECT (skipped)")
    

    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE: {n} rows processed")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
