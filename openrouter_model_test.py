import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import HR_PROMPT, MASKING_PROMPT, REWRITING_PROMPT, SECURITY_PROMPT


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODELS = [
    # "openai/gpt-4o-mini",
    # "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-oss-20b",
    # "meta-llama/llama-3.2-1b-instruct",
    # "google/gemma-2b-it", 
]

DATASET_PATH = Path(__file__).parent / "hiring_v1.csv"
LOG_PATH = Path(__file__).parent / "experiment_logs"
TEMPERATURE = 0.2
MAX_TOKENS = 1024

LOG_PATH.mkdir(exist_ok=True)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


def call_openrouter(prompt: str, model: str) -> str:
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logging.error(f"Error calling OpenRouter with model {model}: {exc}")
        return ""


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()

        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError as exc:
        logging.error(f"Failed to parse JSON response: {exc}")
        logging.error(f"Raw response length: {len(response)}")
        logging.error(f"Raw response sample: {repr(response[:200])}")
        return None
    except Exception as exc:
        logging.error(f"Error parsing response: {exc}")
        return None


def parse_security_json_response(response: str) -> Optional[Dict[str, Any]]:
    parsed = parse_json_response(response)
    if not parsed:
        return None
    keycard_access = parsed.get("keycard_access")
    if keycard_access is not None and not isinstance(keycard_access, dict):
        logging.warning("keycard_access must be a dictionary")
        return None
    return parsed


def run_experiment_for_model(model: str) -> None:
    model_safe_name = model.replace(":", "_").replace("/", "_")
    csv_out_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH / f"all_model_{model_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting HR data extraction experiment with model: {model}")

    try:
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Loaded dataset with {len(df)} entries")
    except Exception as exc:
        logging.error(f"Failed to load dataset: {exc}")
        return

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        general_message = row["general_message"]
        logging.info(f"Processing message {idx + 1}/{len(df)}: {general_message[:100]}...")

        extraction_prompt = HR_PROMPT.format(message=general_message)
        extraction_start = time.time()
        extraction_response = call_openrouter(extraction_prompt, model)
        extraction_end = time.time()
        extracted_data = parse_json_response(extraction_response) or {}

        masking_prompt = MASKING_PROMPT.format(message=general_message)
        masked_message = call_openrouter(masking_prompt, model)
        if not masked_message:
            logging.warning(f"Masking failed for message {idx}; using original text")
            masked_message = general_message

        rewriting_prompt = REWRITING_PROMPT.format(masked_message=masked_message)
        security_message = call_openrouter(rewriting_prompt, model)

        security_prompt = SECURITY_PROMPT.format(message=security_message)
        security_start = time.time()
        security_response = call_openrouter(security_prompt, model)
        security_end = time.time()
        security_data = parse_security_json_response(security_response) or {}

        general_security_prompt = SECURITY_PROMPT.format(message=general_message)
        general_security_start = time.time()
        general_security_response = call_openrouter(general_security_prompt, model)
        general_security_end = time.time()
        general_security_data = parse_security_json_response(general_security_response) or {}

        results.append(
            {
                "model": model,
                "message_id": idx,
                "general_message": general_message,
                "extracted_json": json.dumps(extracted_data),
                "masked_message": masked_message,
                "security_message": security_message,
                "security_json": json.dumps(security_data),
                "general_security_json": json.dumps(general_security_data),
                "extraction_response_time": extraction_end - extraction_start,
                "security_response_time": security_end - security_start,
                "general_security_response_time": general_security_end - general_security_start,
                "timestamp": datetime.now().isoformat(),
            }
        )

    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_out_path, index=False)
        logging.info(f"Results saved to {csv_out_path}")

        print("\n" + "=" * 50)
        print(f"EXPERIMENT SUMMARY FOR {model.upper()}")
        print("=" * 50)
        print(f"\nModel: {model}")
        print(f"  Average Extraction Response Time: {results_df['extraction_response_time'].mean():.2f}s")
        print(f"  Average Security Response Time: {results_df['security_response_time'].mean():.2f}s")
        print(f"  Average General Security Response Time: {results_df['general_security_response_time'].mean():.2f}s")
        print(f"  Total Messages Processed: {len(results_df)}")
        print(f"  Results saved to: {csv_out_path}")
    except Exception as exc:
        logging.error(f"Failed to save results for model {model}: {exc}")


def run_all_models(models: List[str]) -> None:
    print("Starting OpenRouter model testing...")
    print(f"Models to test: {', '.join(models)}")
    print("=" * 60)

    for index, model in enumerate(models, start=1):
        print(f"\n[{index}/{len(models)}] Testing model: {model}")
        print("-" * 40)
        try:
            run_experiment_for_model(model)
            print(f"✅ Completed testing for {model}")
        except Exception as exc:
            print(f"❌ Error testing {model}: {exc}")
            logging.error(f"Failed to test model {model}: {exc}")
        if index < len(models):
            print("Waiting 5 seconds before the next model...")
            time.sleep(5)

    print("\n" + "=" * 60)
    print("OPENROUTER MODEL TESTING COMPLETED")
    print("=" * 60)
    for model in models:
        model_safe_name = model.replace(":", "_").replace("/", "_")
        csv_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"
        print(f"  - {model}: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HR agent experiment against OpenRouter models.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of OpenRouter model identifiers to evaluate.",
    )
    return parser.parse_args()


def main(models: Optional[List[str]] = None) -> None:
    selected_models = models or DEFAULT_MODELS
    if not selected_models:
        raise ValueError("At least one model must be provided.")
    run_all_models(selected_models)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.models)
