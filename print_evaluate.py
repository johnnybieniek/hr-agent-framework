#!/usr/bin/env python3
"""
Detailed evaluator for HR-agent results that prints expected vs received for each field.

- Loads ground truth from hiring_v1.csv
- Evaluates HR JSON extraction (name, position, salary, location, start_date)
- Evaluates Security JSON extraction (name, position, security_level, keycard_access)
- Evaluates masking (salary number and location must not appear in masked_message)
- Prints detailed expected vs received for each field with correct/wrong labels

Usage:
  python print_evaluate.py RESULTS_CSV [--limit N]

Behavior with --limit N:
  - Evaluates only first N rows
  - Prints expected vs received for each of those rows
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    return norm_str(v)


REMOTE_GENERIC_REGION_RAW = [
    "global",
    "worldwide",
    "anywhere",
    "international",
    "all regions",
    "all region",
    "all locations",
    "any location",
    "any timezone",
    "any time zone",
]


def _normalize_region_label(region: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", region.lower())


REMOTE_GENERIC_REGION_TOKENS = {_normalize_region_label(token) for token in REMOTE_GENERIC_REGION_RAW}

REMOTE_REGION_PATTERNS = [
    re.compile(r"remote\s*\(([^)]+)\)", re.IGNORECASE),
    re.compile(r"remote\s*[-–—]\s*([A-Za-z0-9 /,&()]+?)(?=$|\n|[.;!,])", re.IGNORECASE),
    re.compile(r"remote\s+(?:based\s+)?in\s+([A-Za-z0-9 /,&()]+?)(?=$|\n|[.;!,])", re.IGNORECASE),
]


def _extract_remote_region(text: str) -> str:
    for pattern in REMOTE_REGION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" ,.:-")
    return ""


def remote_descriptor(loc: Any) -> Tuple[str, str]:
    text = norm_str(loc)
    if not text:
        return ("non_remote", "")
    if "remote" not in text.lower():
        return ("non_remote", "")
    region = _extract_remote_region(text)
    norm_region = _normalize_region_label(region) if region else ""
    if norm_region in REMOTE_GENERIC_REGION_TOKENS:
        region = ""
        norm_region = ""
    remote_type = "remote_specific" if region else "remote_generic"
    return (remote_type, norm_region)


def positions_match(gt_pos: Any, pred_pos: Any) -> bool:
    gt_norm = norm_str_ci(gt_pos)
    pred_norm = norm_str_ci(pred_pos)
    if not gt_norm:
        return pred_norm == ""
    if not pred_norm:
        return False
    if gt_norm == pred_norm:
        return True

    def strip_department(text: str) -> str:
        return re.sub(r"\s+in\s+the\s+.+?\s+department$", "", text).strip()

    if strip_department(pred_norm) == gt_norm:
        return True
    if strip_department(gt_norm) == pred_norm:
        return True

    if pred_norm.startswith(gt_norm) and pred_norm[len(gt_norm):len(gt_norm)+1] in {"", " ", ",", "-", "(", "/"}:
        return True
    if gt_norm.startswith(pred_norm) and gt_norm[len(pred_norm):len(pred_norm)+1] in {"", " ", ",", "-", "(", "/"}:
        return True

    return False


def _location_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in re.split(r",|\s+-\s+", text) if token.strip()]


def locations_match(gt_loc: Any, pred_loc: Any) -> bool:
    gt_raw = norm_str(gt_loc)
    pred_raw = norm_str(pred_loc)
    if not gt_raw:
        return pred_raw == ""
    if not pred_raw:
        return False
    gt_norm = norm_str_ci(gt_raw)
    pred_norm = norm_str_ci(pred_raw)
    if gt_norm == pred_norm:
        return True
    gt_type, gt_region = remote_descriptor(gt_raw)
    pred_type, pred_region = remote_descriptor(pred_raw)
    if gt_type == "remote_generic" and pred_type == "remote_generic":
        return True
    if gt_type == "remote_specific" and pred_type == "remote_specific":
        return gt_region != "" and gt_region == pred_region
    if gt_type == pred_type == "non_remote":
        gt_tokens = _location_tokens(gt_norm)
        pred_tokens = _location_tokens(pred_norm)
        if not gt_tokens or not pred_tokens:
            return gt_norm == pred_norm
        longer, shorter = (gt_tokens, pred_tokens) if len(gt_tokens) >= len(pred_tokens) else (pred_tokens, gt_tokens)
        if shorter == longer[: len(shorter)]:
            return True
    return False



def gt_keycard_from_row(row) -> Dict[str, int]:
    def access_helper(col: str) -> int:
        val = row.get(col) 
        if pd.isna(val) or val == "":
            return 0 # treat missing/blank access 
        try: 
            return int(val) 
        except:
            return 0
        
    return {
        "Europe": access_helper("access_europe"),
        "North America": access_helper("access_north_america"),
        "South America": access_helper("access_south_america"),
        "Africa": access_helper("access_africa"),
        "Asia": access_helper("access_asia"), 
        "Oceania": access_helper("access_oceania")
    }


def eval_hr_row(gt_row, result_row) -> Dict[str, bool]:
    try:
        data = json.loads(result_row["extracted_json"]) if result_row["extracted_json"] else {}
    except Exception:
        data = {}

    gt_name = f"{gt_row['first_name']} {gt_row['last_name']}"
    name_ok = norm_str_ci(data.get("name")) == norm_str_ci(gt_name)

    if pd.notna(gt_row["position"]):
        pos_ok = positions_match(gt_row["position"], data.get("position"))
    else:
        pos_ok = norm_str_ci(data.get("position")) == ""

    sal_ok = norm_int(data.get("salary")) == norm_int(gt_row["salary"])

    if pd.notna(gt_row["location"]):
        loc_ok = locations_match(gt_row["location"], data.get("location"))
    else:
        loc_ok = norm_str_ci(data.get("location")) == ""

    date_ok = norm_date(data.get("start_date")) == norm_date(gt_row["start_date"]) if pd.notna(gt_row["start_date"]) else norm_date(data.get("start_date")) == ""

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
    pos_ok = norm_str_ci(data.get("position")) == norm_str_ci(gt_row["position"]) if pd.notna(gt_row["position"]) else norm_str_ci(data.get("position")) == ""
    level_val = norm_int(data.get("security_level"))
    level_ok = level_val is not None and level_val in {1, 2, 3}

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
    pos_ok = norm_str_ci(data.get("position")) == norm_str_ci(gt_row["position"]) if pd.notna(gt_row["position"]) else norm_str_ci(data.get("position")) == ""
    level_val = norm_int(data.get("security_level"))
    level_ok = level_val is not None and level_val in {1, 2, 3}

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

    sal = norm_str(gt_row["salary"])  # numeric as string
    if sal == "":
        sal_ok = True
    else:
        sal_ok = (sal not in masked) and (f"${sal}" not in masked)

    loc = norm_str(gt_row["location"]).lower()
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
    
    gt_pos = gt_row["position"] if pd.notna(gt_row["position"]) else ""
    hr_pos = hr_data.get("position", "")
    pos_ok = positions_match(gt_pos, hr_pos)
    
    gt_sal = gt_row["salary"]
    hr_sal = hr_data.get("salary")
    hr_sal_ok = norm_int(hr_sal) == norm_int(gt_sal)
    
    gt_loc = gt_row["location"] if pd.notna(gt_row["location"]) else ""
    hr_loc = hr_data.get("location", "")
    hr_loc_ok = locations_match(gt_loc, hr_loc)
    
    gt_date = gt_row["start_date"] if pd.notna(gt_row["start_date"]) else ""
    hr_date = hr_data.get("start_date", "")
    date_ok = norm_date(hr_date) == norm_date(gt_date)
    
    # Security JSON evaluation (masked)
    try:
        sec_masked_data = json.loads(res_row["security_json"]) if res_row["security_json"] else {}
    except Exception:
        sec_masked_data = {}
    
    sec_name_ok = norm_str_ci(sec_masked_data.get("name", "")) == norm_str_ci(gt_name)
    sec_pos_ok = positions_match(gt_pos, sec_masked_data.get("position", ""))
    sec_level = sec_masked_data.get("security_level")
    sec_level_val = norm_int(sec_level)
    sec_level_ok = sec_level_val is not None and sec_level_val in {1, 2, 3}
    
    gt_key = gt_keycard_from_row(gt_row)
    pred_key = sec_masked_data.get("keycard_access", {}) or {}
    sec_key_ok = all(int(pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(pred_key) >= 6
    
    # Security JSON evaluation (general)
    try:
        sec_general_data = json.loads(res_row["general_security_json"]) if res_row.get("general_security_json") else {}
    except Exception:
        sec_general_data = {}
    
    sec_gen_name_ok = norm_str_ci(sec_general_data.get("name", "")) == norm_str_ci(gt_name)
    sec_gen_pos_ok = positions_match(gt_pos, sec_general_data.get("position", ""))
    sec_gen_level = sec_general_data.get("security_level")
    sec_gen_level_val = norm_int(sec_gen_level)
    sec_gen_level_ok = sec_gen_level_val is not None and sec_gen_level_val in {1, 2, 3}
    
    sec_gen_pred_key = sec_general_data.get("keycard_access", {}) or {}
    sec_gen_key_ok = all(int(sec_gen_pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(sec_gen_pred_key) >= 6
    
    # Masking evaluation
    masked = res_row["masked_message"] or ""
    sal = norm_str(gt_row["salary"])
    if sal == "":
        mask_sal_ok = True
    else:
        mask_sal_ok = (sal not in masked) and (f"${sal}" not in masked)
    
    loc = norm_str(gt_row["location"]).lower()
    mask_loc_ok = (loc == "") or (loc not in masked.lower())
    
    # Return True if ANY field has an error
    return not all([
        name_ok, pos_ok, hr_sal_ok, hr_loc_ok, date_ok,  # HR fields
        sec_name_ok, sec_pos_ok, sec_level_ok, sec_key_ok,  # Security masked
        sec_gen_name_ok, sec_gen_pos_ok, sec_gen_level_ok, sec_gen_key_ok,  # Security general
        mask_sal_ok, mask_loc_ok  # Masking
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
    gt_pos = gt_row["position"] if pd.notna(gt_row["position"]) else ""
    hr_pos = hr_data.get("position", "")
    pos_ok = positions_match(gt_pos, hr_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{hr_pos}' | {'✅ CORRECT' if pos_ok else '❌ WRONG'}")
    
    # Salary
    gt_sal = gt_row["salary"]
    hr_sal = hr_data.get("salary")
    sal_ok = norm_int(hr_sal) == norm_int(gt_sal)
    print(f"Salary:   Expected='{gt_sal}' | Received='{hr_sal}' | {'✅ CORRECT' if sal_ok else '❌ WRONG'}")
    
    # Location
    gt_loc = gt_row["location"] if pd.notna(gt_row["location"]) else ""
    hr_loc = hr_data.get("location", "")
    loc_ok = locations_match(gt_loc, hr_loc)
    print(f"Location: Expected='{gt_loc}' | Received='{hr_loc}' | {'✅ CORRECT' if loc_ok else '❌ WRONG'}")
    
    # Start Date
    gt_date = gt_row["start_date"] if pd.notna(gt_row["start_date"]) else ""
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
    sec_pos_ok = positions_match(gt_pos, sec_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{sec_pos}' | {'✅ CORRECT' if sec_pos_ok else '❌ WRONG'}")
    
    # Security Level
    sec_level = sec_masked_data.get("security_level")
    sec_level_val = norm_int(sec_level)
    sec_level_ok = sec_level_val is not None and sec_level_val in {1, 2, 3}
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
    sec_gen_pos_ok = positions_match(gt_pos, sec_gen_pos)
    print(f"Position: Expected='{gt_pos}' | Received='{sec_gen_pos}' | {'✅ CORRECT' if sec_gen_pos_ok else '❌ WRONG'}")
    
    # Security Level
    sec_gen_level = sec_general_data.get("security_level")
    sec_gen_level_val = norm_int(sec_gen_level)
    sec_gen_level_ok = sec_gen_level_val is not None and sec_gen_level_val in {1, 2, 3}
    print(f"Level:    Expected='1/2/3' | Received='{sec_gen_level}' | {'✅ CORRECT' if sec_gen_level_ok else '❌ WRONG'}")
    
    # Keycard Access
    sec_gen_pred_key = sec_general_data.get("keycard_access", {}) or {}
    sec_gen_key_ok = all(int(sec_gen_pred_key.get(k, 0)) == int(v) for k, v in gt_key.items()) and len(sec_gen_pred_key) >= 6
    print(f"Keycard:  Expected='{gt_key}' | Received='{sec_gen_pred_key}' | {'✅ CORRECT' if sec_gen_key_ok else '❌ WRONG'}")
    
    # Masking evaluation
    print("\n🎭 MASKING EVALUATION:")
    print("-" * 40)
    
    masked = res_row["masked_message"] or ""
    sal = norm_str(gt_row["salary"])
    print("Salary: ", sal)
    if sal == "":
        sal_found = False
        sal_ok = True
    else:
        sal_found = (sal in masked) or (f"${sal}" in masked)
        sal_ok = not sal_found
    print(f"Salary:   Expected='NOT in masked' | Found='{sal_found}' | {'✅ CORRECT' if sal_ok else '❌ WRONG'}")
    
    loc = norm_str(gt_row["location"]).lower()
    print("Location: ", loc)
    loc_ok = (loc == "") or (loc not in masked.lower())
    print(f"Location: Expected='NOT in masked' | Found='{loc in masked.lower()}' | {'✅ CORRECT' if loc_ok else '❌ WRONG'}")

    print("Masked message: ", masked)


def main():
    p = argparse.ArgumentParser(description="Detailed evaluation of HR-Agent results")
    p.add_argument("results", help="Path to results CSV (e.g., 3agent_xxx_results.csv)")
    p.add_argument("--ground-truth", default="main_dataset.csv", help="Path to ground truth CSV")
    p.add_argument("--limit", type=int, default=1000, help="Evaluate only first N rows (default: 100)")
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
def _location_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in re.split(r",|\s+-\s+", text) if token.strip()]
