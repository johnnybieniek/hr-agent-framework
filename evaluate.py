#!/usr/bin/env python3
"""
Simple evaluator for HR-agent results that displays final summary statistics.

- Loads ground truth from main_dataset.csv (default)
- Evaluates HR JSON extraction (name, position, salary, location, start_date)
- Evaluates Security JSON extraction (name, position, security_level, keycard_access)
- Evaluates masking (salary number and location must not appear in masked_message)
- Displays final accuracy statistics only (no per-row details)

Usage:
  python evaluate.py RESULTS_CSV [--limit N]
  Llama: python3 evaluate.py 3agent_llama3.1_8b_results.csv
  gemma: python3 evaluate.py 3agent_gemma3n_e4b_results.csv
  gpt-5-mini: python3 evaluate.py 3agent_gpt-5-mini_results.csv
  llama-3.1-8b-instant: python3 evaluate.py 3agent_llama-3.1-8b-instant_results.csv
  llama-3.3-70b-versatile: python3 evaluate.py 3agent_llama-3.3-70b-versatile_results.csv
  openai/gpt-oss-20b: python3 evaluate.py 3agent_openai_gpt-oss-20b_results.csv
  meta-llama/llama-3.1-8b-instruct: python3 evaluate.py 3agent_meta-llama_llama-3.1-8b-instruct_results.csv

Behavior with --limit N:
  - Evaluates only first N rows
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
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


def _location_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in re.split(r",|\s+-\s+", text) if token.strip()]


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


def main():
    p = argparse.ArgumentParser(description="Evaluate HR-Agent results")
    p.add_argument("results", help="Path to results CSV (e.g., 3agent_xxx_results.csv)")
    p.add_argument("--ground-truth", default="main_dataset.csv", help="Path to ground truth CSV")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows")
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

    print(f"📊 Ground truth: {len(gt_df)} rows")
    print(f"📊 Results: {len(res_df)} rows")
    print(f"📊 Results message_id range: {res_df['message_id'].min()} to {res_df['message_id'].max()}")

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
    

    for i in range(n):
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

    total = n
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
