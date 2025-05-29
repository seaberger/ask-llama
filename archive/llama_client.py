#!/usr/bin/env python3
"""
llama_client.py - Demonstration script for interacting with a local llama-server
using the OpenAI Python client library.

This script shows how to configure the OpenAI client to work with a locally running
llama-server instance, and demonstrates both chat completions and text completions.
"""

import sys
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union
import os

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI Python package not found. Installing it now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai
    from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set constants for the server
LLAMA_SERVER_URL = "http://localhost:8080"
MODEL_ID = "/Users/seanbergman/Library/Caches/llama.cpp/ggml-org_Qwen2.5-Coder-3B-Q8_0-GGUF_qwen2.5-coder-3b-q8_0.gguf"

def configure_client() -> OpenAI:
    """
    Configure and return an OpenAI client instance that connects to the local llama-server.
    
    Returns:
        OpenAI: Configured client instance
    """
    # Create the client instance with the base URL pointing to our local server
    # No API key is required for the local server
    client = OpenAI(
        base_url=LLAMA_SERVER_URL,
        api_key="not-needed",  # llama-server doesn't require API key by default
        timeout=60.0  # Increase timeout for longer generations
    )
    logging.info("OpenAI client configured with base URL: %s", LLAMA_SERVER_URL)
    return client

def get_chat_completion(
    client: OpenAI, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Send a chat completion request to the llama-server.
    
    Args:
        client: Configured OpenAI client
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (None for model default)
        
    Returns:
        str: The generated response text
    """
    try:
        logging.info("Sending chat completion request with %d messages", len(messages))
        logging.debug("Messages: %s", json.dumps(messages, indent=2))
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Log the raw response for debugging
        logging.debug("Raw chat completion response: %s", response)
        
        # Extract and return the content
        content = response.choices[0].message.content
        logging.info("Received chat completion response (%d characters)", len(content))
        return content
    except openai.APIError as e:
        logging.error("OpenAI API error: %s", str(e))
        return f"API Error: {str(e)}"
    except openai.APIConnectionError as e:
        logging.error("Failed to connect to OpenAI API: %s", str(e))
        return f"Connection Error: {str(e)}"
    except openai.RateLimitError as e:
        logging.error("Rate limit exceeded: %s", str(e))
        return f"Rate Limit Error: {str(e)}"
    except Exception as e:
        logging.error("Unexpected error in chat completion: %s", str(e), exc_info=True)
        return f"Error: {str(e)}"

def get_text_completion(
    client: OpenAI, 
    prompt: str, 
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Send a text completion request to the llama-server.
    
    Args:
        client: Configured OpenAI client
        prompt: The text prompt to complete
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (None for model default)
        
    Returns:
        str: The generated completion text
    """
    try:
        logging.info("Sending text completion request with prompt length %d chars", len(prompt))
        logging.debug("Prompt: %s", prompt)
        
        response = client.completions.create(
            model=MODEL_ID,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Log the raw response for debugging
        logging.debug("Raw text completion response: %s", response)
        
        # Extract and return the text
        text = response.choices[0].text
        logging.info("Received text completion response (%d characters)", len(text))
        return text
    except openai.APIError as e:
        logging.error("OpenAI API error: %s", str(e))
        return f"API Error: {str(e)}"
    except openai.APIConnectionError as e:
        logging.error("Failed to connect to OpenAI API: %s", str(e))
        return f"Connection Error: {str(e)}"
    except openai.RateLimitError as e:
        logging.error("Rate limit exceeded: %s", str(e))
        return f"Rate Limit Error: {str(e)}"
    except Exception as e:
        logging.error("Unexpected error in text completion: %s", str(e), exc_info=True)
        return f"Error: {str(e)}"

def get_available_models(client: OpenAI) -> List[Dict[str, Any]]:
    """
    Get a list of available models from the server.
    
    Args:
        client: Configured OpenAI client
        
    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
    try:
        logging.info("Fetching available models from server")
        response = client.models.list()
        logging.debug("Raw models response: %s", response)
        logging.info("Found %d models", len(response.data))
        return response.data
    except openai.APIError as e:
        logging.error("OpenAI API error when fetching models: %s", str(e))
        return []
    except openai.APIConnectionError as e:
        logging.error("Failed to connect to OpenAI API when fetching models: %s", str(e))
        return []
    except Exception as e:
        logging.error("Unexpected error when fetching models: %s", str(e), exc_info=True)
        return []

def check_server_connection() -> bool:
    """
    Check if the llama-server is accessible and responding.
    
    Returns:
        bool: True if server is accessible, False otherwise
    """
    try:
        logging.info("Testing connection to llama-server at %s", LLAMA_SERVER_URL)
        response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        
        if response.status_code == 200:
            logging.info("Successfully connected to llama-server (status code: %d)", response.status_code)
            return True
        else:
            logging.error("Server responded with status code: %d", response.status_code)
            logging.error("Response body: %s", response.text)
            return False
    except requests.RequestException as e:
        logging.error("Failed to connect to llama-server: %s", str(e))
        return False

def main():
    """Main function demonstrating the use of local llama-server."""
    print("\n===== LLAMA-SERVER CLIENT DEMO =====", flush=True)
    
    # Step 1: Test server connectivity
    if not check_server_connection():
        print("\nERROR: Cannot connect to llama-server at", LLAMA_SERVER_URL, flush=True)
        print("Please check if the server is running and try again.", flush=True)
        sys.exit(1)
    
    # Step 2: Configure the client
    print("\nConfiguring OpenAI client for local llama-server...", flush=True)
    client = configure_client()
    
    # Step 3: Check available models
    print("\nFetching available models...", flush=True)
    models = get_available_models(client)
    if models:
        print(f"Found {len(models)} model(s):", flush=True)
        for model in models:
            print(f" - {model.id}", flush=True)
    else:
        print("No models found or error occurred.", flush=True)
        sys.exit(1)
    
    # Step 4: Simple test ping with shorter prompt
    print("\n=== Quick Test Ping ===", flush=True)
    test_messages = [
        {"role": "user", "content": "Say hello in one short sentence."}
    ]
    print("Sending quick test message...", flush=True)
    start_time = time.time()
    test_response = get_chat_completion(client, test_messages)
    elapsed = time.time() - start_time
    print(f"\nTest response received in {elapsed:.2f} seconds:", flush=True)
    print(test_response, flush=True)
    
    # Continue only if quick test was successful
    if "Error" in test_response:
        print("\nQuick test failed. Stopping execution.", flush=True)
        sys.exit(1)
    
    # Step 5: Chat completion example
    print("\n=== Chat Completion Example ===", flush=True)
    chat_messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence."}
    ]
    print("Sending chat completion request...", flush=True)
    start_time = time.time()
    chat_response = get_chat_completion(client, chat_messages)
    elapsed = time.time() - start_time
    print(f"\nResponse received in {elapsed:.2f} seconds:", flush=True)
    print(chat_response, flush=True)
    
    # Step 6: Text completion example
    print("\n=== Text Completion Example ===", flush=True)
    prompt = "Python function to sort a list of numbers:"
    print(f"Sending text completion request with prompt: '{prompt}'", flush=True)
    start_time = time.time()
    completion_response = get_text_completion(client, prompt)
    elapsed = time.time() - start_time
    print(f"\nResponse received in {elapsed:.2f} seconds:", flush=True)
    print(completion_response, flush=True)
    
    print("\n===== DEMO COMPLETED SUCCESSFULLY =====", flush=True)

if __name__ == "__main__":
    main()

