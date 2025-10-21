import json
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
from openai import OpenAI
# Importing all of the prompts from a separate file
from prompts import HR_PROMPT, MASKING_PROMPT, REWRITING_PROMPT, SECURITY_PROMPT

# Ensure environment variables from .env are loaded before reading API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
_groq_client: Optional[OpenAI] = None

def get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client

# Array of models to test (Groq models)
MODELS = [
    #"llama-3.3-70b-versatile",
    #"llama-3.1-8b-instant", 
    #"mixtral-8x7b-32768",
    "openai/gpt-oss-20b",
    #"gemma-7b-it"
    ]

DATASET_PATH = Path(__file__).parent / "hiring_v1.csv"
LOG_PATH = Path(__file__).parent / "experiment_logs"

# Create logs directory if it doesn't exist
LOG_PATH.mkdir(exist_ok=True)


def call_groq_chat(prompt: str, model: str) -> str:
    """Call Groq API and return the assistant message content."""
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY is not set.")
        return ""

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return ""


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from the model, handling common formatting issues."""
    try:
        # Trying to extract JSON if it's wrapped in markdown code blocks - often happens with gemma
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
        logging.error(f"Raw response length: {len(response)}")
        logging.error(f"Raw response: {repr(response[:200])}...")  # Show first 200 chars
        return None
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        return None


def parse_security_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from the security agent, handling common formatting issues."""
    try:
        # Same logic as in the parse_json_response function
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
        
        if not isinstance(parsed.get("keycard_access"), dict):
            logging.warning("keycard_access must be a dictionary")
            return None
        
        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse security JSON response: {e}")
        logging.error(f"Raw security response length: {len(response)}")
        logging.error(f"Raw security response: {repr(response[:200])}...")  # Show first 200 chars
        return None
    except Exception as e:
        logging.error(f"Error parsing security response: {e}")
        return None


def run_experiment_for_model(model: str):
    """Run the experiment for a single model."""
    # Creating model-specific file paths
    model_safe_name = model.replace(":", "_").replace("/", "_")
    csv_out_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"
    
    # Setting up model-specific logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH / f"all_model_{model_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting HR data extraction experiment with model: {model}")
    
    try:
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Loaded dataset with {len(df)} entries")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    results = []
    
    for idx, row in df.iterrows():
        general_message = row['general_message']
        logging.info(f"Processing message {idx + 1}/{len(df)}: {general_message[:100]}...")
        
        # ------------------------------------------------------------------
        # Step 1: Extraction agent
        # Format and send the extraction prompt to the model, then parse the
        # returned JSON into a Python dict.
        extraction_prompt = HR_PROMPT.format(message=general_message)
        start_time = time.time()
        extraction_response = call_groq_chat(extraction_prompt, model)
        end_time = time.time()
        extracted_data = parse_json_response(extraction_response) or {}

        # Step 2: Masking agent
        # Use the imported masking prompt to remove salary and generalize
        # location.  The agent operates on the raw message and returns
        # an edited message with the sensitive information masked.
        masking_prompt = MASKING_PROMPT.format(message=general_message)
        masked_message = call_groq_chat(masking_prompt, model)
        
        # Fallback: if masking fails, use original message
        if not masked_message or masked_message.strip() == "":
            logging.warning(f"Masking failed for message {idx}, using original message")
            masked_message = general_message

        # Step 3: Rewriting agent
        # Use the imported rewriting prompt to craft the final message for
        # the security team based on the masked message.
        rewriting_prompt = REWRITING_PROMPT.format(masked_message=masked_message)
        security_message = call_groq_chat(rewriting_prompt, model)

        # Step 4: Security agent with masked message
        # Use the imported security prompt to analyze the security message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=security_message)
        security_start_time = time.time()
        security_response = call_groq_chat(security_prompt, model)
        security_end_time = time.time()
        security_data = parse_security_json_response(security_response) or {}

        # Step 5: Security agent with general message
        # Use the imported security prompt to analyze the general message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=general_message)
        general_security_start_time = time.time()
        general_security_response = call_groq_chat(security_prompt, model)
        general_security_end_time = time.time()
        general_security_data = parse_security_json_response(general_security_response) or {}

        # Collect the results for this record.  We include the extracted JSON,
        # the masked message, the security message, and the security JSON.
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

        # Removed per-iteration delay to improve throughput
    
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_out_path, index=False)
        logging.info(f"Results saved to {csv_out_path}")
        
        # Summary statistics
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
        
        # adding delay between models
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
