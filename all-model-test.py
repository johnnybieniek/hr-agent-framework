import json
import httpx
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Importing all of the prompts from a separate file
from prompts import HR_PROMPT, MASKING_PROMPT, REWRITING_PROMPT, SECURITY_PROMPT


OLLAMA_URL = "http://localhost:11434/api/generate"

# Array of models to test
MODELS = [
    "llama3.1:8b",
    "llama3:8b", 
    "mistral:7b",
    "gemma3n:e4b"
]

DATASET_PATH = Path(__file__).parent / "dataset-v2.csv"
LOG_PATH = Path(__file__).parent / "experiment_logs"

# Create logs directory if it doesn't exist
LOG_PATH.mkdir(exist_ok=True)


def call_ollama(prompt: str, model: str) -> str:
    """Call Ollama API and return the response."""
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        logging.error(f"Error calling Ollama with model {model}: {e}")
        return ""


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from the model, handling common formatting issues."""
    try:
        # Try to extract JSON if it's wrapped in markdown code blocks - often happens with gemma
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
        
        # Parse the JSON
        parsed = json.loads(response)
    
        
        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e}")
        logging.error(f"Raw response: {response}")
        return None
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        return None


def parse_security_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from the security agent, handling common formatting issues."""
    try:
        # Try to extract JSON if it's wrapped in markdown code blocks
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
        
        # Parse the JSON
        parsed = json.loads(response)
        
        # Validate keycard_access structure
        if not isinstance(parsed.get("keycard_access"), dict):
            logging.warning("keycard_access must be a dictionary")
            return None
        
        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse security JSON response: {e}")
        logging.error(f"Raw security response: {response}")
        return None
    except Exception as e:
        logging.error(f"Error parsing security response: {e}")
        return None


def run_experiment_for_model(model: str):
    """Run the experiment for a single model."""
    # Create model-specific file paths
    model_safe_name = model.replace(":", "_").replace("/", "_")
    csv_out_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"
    
    # Set up model-specific logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH / f"all_model_{model_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting HR data extraction experiment with model: {model}")
    
    # Load dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Loaded dataset with {len(df)} entries")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    # Prepare results storage
    results = []
    
    # Process each row in the dataset
    for idx, row in df.iterrows():
        general_message = row['general_message']
        logging.info(f"Processing message {idx + 1}/{len(df)}: {general_message[:100]}...")
        
        # ------------------------------------------------------------------
        # Step 1: Extraction agent
        # Format and send the extraction prompt to the model, then parse the
        # returned JSON into a Python dict.
        extraction_prompt = HR_PROMPT.format(message=general_message)
        start_time = time.time()
        extraction_response = call_ollama(extraction_prompt, model)
        end_time = time.time()
        extracted_data = parse_json_response(extraction_response) or {}

        # Step 2: Masking agent
        # Use the imported masking prompt to remove salary and generalize
        # location.  The agent operates on the raw message and returns
        # an edited message with the sensitive information masked.
        masking_prompt = MASKING_PROMPT.format(message=general_message)
        masked_message = call_ollama(masking_prompt, model)

        # Step 3: Rewriting agent
        # Use the imported rewriting prompt to craft the final message for
        # the security team based on the masked message.
        rewriting_prompt = REWRITING_PROMPT.format(masked_message=masked_message)
        security_message = call_ollama(rewriting_prompt, model)

        # Step 4: Security agent with masked message
        # Use the imported security prompt to analyze the security message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=security_message)
        security_start_time = time.time()
        security_response = call_ollama(security_prompt, model)
        security_end_time = time.time()
        security_data = parse_security_json_response(security_response) or {}

        # Step 5: Security agent with general message
        # Use the imported security prompt to analyze the general message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=general_message)
        general_security_start_time = time.time()
        general_security_response = call_ollama(security_prompt, model)
        general_security_end_time = time.time()
        general_security_data = parse_security_json_response(general_security_response) or {}

        # Collect the results for this record.  We include the extracted JSON,
        # the masked message, the security message, and the security JSON.
        # Response times track both extraction and security agent calls.
        result = {
            "model": model,
            "message_id": idx,
            "general_message": general_message,
            "extracted_json": json.dumps(extracted_data),
            "masked_message": masked_message,
            "security_message": security_message,
            "security_json": json.dumps(security_data),
            "general_security_json": json.dumps(general_security_data),
            "extraction_response_time": end_time - start_time,
            "security_response_time": security_end_time - security_start_time,
            "general_security_response_time": general_security_end_time - general_security_start_time,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)

        # Optional delay to avoid overwhelming the API service
        time.sleep(1)
    
    # Save results to CSV
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_out_path, index=False)
        logging.info(f"Results saved to {csv_out_path}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print(f"EXPERIMENT SUMMARY FOR {model.upper()}")
        print("="*50)
        
        avg_extraction_time = results_df['extraction_response_time'].mean()
        avg_security_time = results_df['security_response_time'].mean()
        avg_general_security_time = results_df['general_security_response_time'].mean()
        print(f"\nModel: {model}")
        print(f"  Average Extraction Response Time: {avg_extraction_time:.2f}s")
        print(f"  Average Security Response Time: {avg_security_time:.2f}s")
        print(f"  Average General Security Response Time: {avg_general_security_time:.2f}s")
        print(f"  Total Messages Processed: {len(results_df)}")
        print(f"  Results saved to: {csv_out_path}")
    except Exception as e:
        logging.error(f"Failed to save results for model {model}: {e}")


def run_all_models():
    """Run experiments for all models in the MODELS array."""
    print("Starting comprehensive model testing...")
    print(f"Models to test: {', '.join(MODELS)}")
    print("="*60)
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Testing model: {model}")
        print("-" * 40)
        
        try:
            run_experiment_for_model(model)
            print(f"✅ Completed testing for {model}")
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
            logging.error(f"Failed to test model {model}: {e}")
        
        # Add a longer delay between models to avoid overwhelming the system
        if i < len(MODELS):
            print(f"Waiting 5 seconds before next model...")
            time.sleep(5)
    
    print("\n" + "="*60)
    print("ALL MODEL TESTING COMPLETED")
    print("="*60)
    print(f"Tested {len(MODELS)} models:")
    for model in MODELS:
        model_safe_name = model.replace(":", "_").replace("/", "_")
        csv_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"
        print(f"  - {model}: {csv_path}")


if __name__ == "__main__":
    run_all_models()
