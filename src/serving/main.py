import os
import json
import logging
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our core modules
from src.core.validator import Validator
from src.core.search_space import Genome

# Setup Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("empas-serving")

app = FastAPI(title="EMPAS Inference Service", version="1.0")

# Global Model Handler
model_handler = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

class GenerationResponse(BaseModel):
    generated_text: str
    latency_ms: float

def load_artifact(artifact_name: str = "balanced"):
    """
    Loads the JSON config and initializes the model with that configuration.
    """
    path = f"./deployment/artifacts/{artifact_name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
        
    logger.info(f"Loading artifact: {artifact_name} (Archetype: {config['archetype']})")
    
    # 1. Initialize Wrapper (Validator class has the logic we need)
    handler = Validator(config['config']['base_model'])
    
    # 2. Parse Genes from Config
    quant_map = config['config']['quantization_map']
    # Sort by integer index of layer name
    sorted_keys = sorted(quant_map.keys(), key=lambda x: int(x.split('_')[1]))
    genes = [quant_map[k] for k in sorted_keys]
    
    # 3. Apply Quantization
    genome = Genome(genes=genes)
    handler.apply_genome(genome)
    
    return handler

@app.on_event("startup")
async def startup_event():
    global model_handler
    try:
        # Load the "Balanced" model by default
        model_handler = load_artifact("balanced")
        logger.info("System Ready. Model loaded on GPU.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/health")
def health_check():
    if model_handler is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    if model_handler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start = time.time()
    
    # Simple generation logic using the underlying HF model
    tokenizer = model_handler.wrapper.tokenizer
    model = model_handler.wrapper.model
    
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model_handler.wrapper.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=request.max_length, 
            do_sample=True, 
            temperature=0.7
        )
        
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end = time.time()
    
    latency = (end - start) * 1000
    return {"generated_text": text, "latency_ms": latency}

if __name__ == "__main__":
    # Dev mode execution
    uvicorn.run(app, host="0.0.0.0", port=8000)