import json
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Importing all of the prompts from a separate file
from prompts import HR_PROMPT, MASKING_PROMPT, REWRITING_PROMPT, SECURITY_PROMPT

# Hugging Face model configurations
# Replace these with the actual Hugging Face model IDs you want to use
HF_MODELS = [
    {
        "name": "llama3.1_8b",
        "model_id": "meta-llama/Llama-3.1-8B", 
        "device": "auto" 
    },
    {
        "name": "gemma3n_e4b", 
        "model_id": "google/gemma-3n-E4B", 
        "device": "auto"
    }
]

DATASET_PATH = Path(__file__).parent / "hiring_v1.csv"
LOG_PATH = Path(__file__).parent / "experiment_logs"

# Create logs directory if it doesn't exist
LOG_PATH.mkdir(exist_ok=True)

# Global variable to store loaded models
loaded_models = {}

def load_hf_model(model_config: Dict[str, str]):
    """Load a Hugging Face model and tokenizer."""
    model_id = model_config["model_id"]
    device = model_config["device"]
    
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    try:
        logging.info(f"Loading model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )
        
        loaded_models[model_id] = pipe
        logging.info(f"Successfully loaded model: {model_id}")
        return pipe
        
    except Exception as e:
        logging.error(f"Failed to load model {model_id}: {e}")
        raise

def call_hf_model(prompt: str, model_config: Dict[str, str]) -> str:
    """Call Hugging Face model and return the response."""
    try:
        # Load model if not already loaded
        pipe = load_hf_model(model_config)
        
        # Generate response
        response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        
        # Extract the generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '').strip()
        else:
            generated_text = str(response).strip()
            
        return generated_text
        
    except Exception as e:
        logging.error(f"Error calling HF model {model_config['model_id']}: {e}")
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
        logging.error(f"Raw response: {response}")
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
        logging.error(f"Raw security response: {response}")
        return None
    except Exception as e:
        logging.error(f"Error parsing security response: {e}")
        return None

def run_experiment_for_model(model_config: Dict[str, str]):
    """Run the experiment for a single model."""
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    
    # Creating model-specific file paths
    model_safe_name = model_name.replace(":", "_").replace("/", "_")
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
    
    logging.info(f"Starting HR data extraction experiment with model: {model_name} ({model_id})")
    
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
        extraction_response = call_hf_model(extraction_prompt, model_config)
        end_time = time.time()
        extracted_data = parse_json_response(extraction_response) or {}

        # Step 2: Masking agent
        # Use the imported masking prompt to remove salary and generalize
        # location.  The agent operates on the raw message and returns
        # an edited message with the sensitive information masked.
        masking_prompt = MASKING_PROMPT.format(message=general_message)
        masked_message = call_hf_model(masking_prompt, model_config)

        # Step 3: Rewriting agent
        # Use the imported rewriting prompt to craft the final message for
        # the security team based on the masked message.
        rewriting_prompt = REWRITING_PROMPT.format(masked_message=masked_message)
        security_message = call_hf_model(rewriting_prompt, model_config)

        # Step 4: Security agent with masked message
        # Use the imported security prompt to analyze the security message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=security_message)
        security_start_time = time.time()
        security_response = call_hf_model(security_prompt, model_config)
        security_end_time = time.time()
        security_data = parse_security_json_response(security_response) or {}

        # Step 5: Security agent with general message
        # Use the imported security prompt to analyze the general message and
        # return security-related JSON data including security level and keycard access.
        security_prompt = SECURITY_PROMPT.format(message=general_message)
        general_security_start_time = time.time()
        general_security_response = call_hf_model(security_prompt, model_config)
        general_security_end_time = time.time()
        general_security_data = parse_security_json_response(general_security_response) or {}

        # Collect the results for this record.  We include the extracted JSON,
        # the masked message, the security message, and the security JSON.
        result = {
            "model": model_name,
            "model_id": model_id,
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

        # Optional delay to avoid overwhelming the system
        time.sleep(1)
    
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_out_path, index=False)
        logging.info(f"Results saved to {csv_out_path}")
        
        # Summary statistics
        print("\n" + "="*50)
        print(f"EXPERIMENT SUMMARY FOR {model_name.upper()}")
        print("="*50)
        
        avg_extraction_time = results_df['extraction_response_time'].mean()
        avg_security_time = results_df['security_response_time'].mean()
        avg_general_security_time = results_df['general_security_response_time'].mean()
        print(f"\nModel: {model_name} ({model_id})")
        print(f"  Average Extraction Response Time: {avg_extraction_time:.2f}s")
        print(f"  Average Security Response Time: {avg_security_time:.2f}s")
        print(f"  Average General Security Response Time: {avg_general_security_time:.2f}s")
        print(f"  Total Messages Processed: {len(results_df)}")
        print(f"  Results saved to: {csv_out_path}")
    except Exception as e:
        logging.error(f"Failed to save results for model {model_name}: {e}")

def run_all_models():
    """Run experiments for all models in the HF_MODELS array."""
    print("Starting comprehensive Hugging Face model testing...")
    print(f"Models to test: {', '.join([m['name'] for m in HF_MODELS])}")
    print("="*60)
    
    for i, model_config in enumerate(HF_MODELS, 1):
        model_name = model_config["name"]
        print(f"\n[{i}/{len(HF_MODELS)}] Testing model: {model_name}")
        print(f"Model ID: {model_config['model_id']}")
        print("-" * 40)
        
        try:
            run_experiment_for_model(model_config)
            print(f"✅ Completed testing for {model_name}")
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            logging.error(f"Failed to test model {model_name}: {e}")
        
        # adding delay between models
        if i < len(HF_MODELS):
            print(f"Waiting 5 seconds before next model...")
            time.sleep(5)
    
    print("\n" + "="*60)
    print("ALL MODEL TESTING COMPLETED")
    print("="*60)
    print(f"Tested {len(HF_MODELS)} models:")
    for model_config in HF_MODELS:
        model_safe_name = model_config["name"].replace(":", "_").replace("/", "_")
        csv_path = Path(__file__).parent / f"3agent_{model_safe_name}_results.csv"
        print(f"  - {model_config['name']}: {csv_path}")

if __name__ == "__main__":
    run_all_models()
