#!/usr/bin/env python3
"""
Script to initialize Florence-2 model with trust_remote_code=True
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize Florence-2 model with trust_remote_code=True")
    parser.add_argument("--model_path", type=str, 
                       default="/Users/seanbergman/.lmstudio/models/ljnlonoljpiljm/florence-2-base-nsfw-v2-ext-mlx",
                       help="Path to the Florence-2 model")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return 1
    
    try:
        logger.info(f"Loading model from {model_path} with trust_remote_code=True")
        
        # Import libraries here to catch any import errors
        try:
            # Import basic libraries first
            import mlx
            import mlx_vlm
            from transformers import AutoProcessor
            import sys
            from importlib.util import spec_from_file_location, module_from_spec
            
            # Log MLX-VLM version
            logger.info(f"MLX-VLM version: {mlx_vlm.__version__}")
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            logger.info("Make sure you've installed the required dependencies with: pip install -U mlx-vlm transformers")
            return 1
        
        # Add the model path to Python's import path so we can import the processor module
        logger.info(f"Adding {model_path} to sys.path to import processing module...")
        sys.path.insert(0, model_path)
        
        # Try to import the Florence-2 processor directly
        try:
            logger.info("Importing Florence-2 processor module...")
            import processing_florence2
            logger.info("Successfully imported Florence-2 processor module")
        except ImportError as e:
            # If direct import fails, try to load it as a module from the path
            logger.info(f"Direct import failed: {e}. Trying alternative loading method...")
            try:
                spec = spec_from_file_location("processing_florence2", 
                                              f"{model_path}/processing_florence2.py")
                processing_florence2 = module_from_spec(spec)
                spec.loader.exec_module(processing_florence2)
                logger.info("Successfully loaded Florence-2 processor module")
            except Exception as e:
                logger.error(f"Failed to load processing_florence2.py: {e}")
                return 1
        
        # Load the processor first
        logger.info("Loading processor with trust_remote_code=True...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            logger.info("Processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            logger.info("This could be due to missing dependencies or incompatible model files.")
            return 1
        
        # Now use the MLX-VLM load function to load the model
        logger.info("Loading Florence-2 model with mlx_vlm.load()...")
        try:
            # Use the mlx_vlm.load() function which is the correct way to load MLX models
            florence_model = mlx_vlm.load(
                model_path,
                trust_remote_code=True
            )
            
            # Validate the model was loaded successfully
            logger.info("Model loaded successfully")
            logger.info(f"Loaded model type: {type(florence_model)}")
            
            # Check if important attributes or methods exist
            if hasattr(florence_model, "generate"):
                logger.info("Generate method found - model appears fully operational")
            else:
                logger.info("Model loaded but doesn't have expected 'generate' method - still may work in LM Studio")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("The model failed to load, check that you have the correct MLX-VLM version")
            return 1

        logger.info("\n✅ Successfully loaded Florence-2 model with trust_remote_code=True")
        logger.info("You can now use this model in LM Studio by selecting this Python environment")
        logger.info("Path to Python: ~/venvs/py3.12-general/bin/python")
        
        # Provide instructions for LM Studio configuration
        logger.info("\nInstructions for LM Studio:")
        logger.info("1. In LM Studio, go to Settings > Python")
        logger.info("2. Set Python Interpreter Path to: /Users/seanbergman/venvs/py3.12-general/bin/python")
        logger.info("3. When loading the Florence-2 model, make sure to check 'Trust remote code'")
        logger.info("4. The model should now load correctly\n")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
