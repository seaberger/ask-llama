#!/usr/bin/env python3
"""
test_llama.py - Simple test script for interacting with a local llama-server

This minimal script tests basic connectivity and functionality with the llama-server.
"""

import sys
import json
import traceback
import time

# Set server constants
LLAMA_SERVER_URL = "http://localhost:8080"
MODEL_ID = "/Users/seanbergman/Library/Caches/llama.cpp/ggml-org_Qwen2.5-Coder-3B-Q8_0-GGUF_qwen2.5-coder-3b-q8_0.gguf"

print("\n===== LLAMA-SERVER TEST SCRIPT =====", flush=True)
print(f"Python version: {sys.version}", flush=True)

# Test 1: Simple requests module test
print("\n----- Test 1: Testing with requests library -----", flush=True)
try:
    print("Importing requests module...", flush=True)
    import requests
    print("✓ Successfully imported requests", flush=True)
    
    print(f"Making GET request to {LLAMA_SERVER_URL}/v1/models...", flush=True)
    response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
    print(f"✓ Server responded with status code: {response.status_code}", flush=True)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {len(data.get('data', []))} models", flush=True)
        for model in data.get('data', []):
            print(f"  - {model.get('id')}", flush=True)
    else:
        print(f"✗ Error response: {response.text}", flush=True)
        
except Exception as e:
    print(f"✗ Error in Test 1: {str(e)}", flush=True)
    print("\nTraceback:", flush=True)
    traceback.print_exc()

# Test 2: OpenAI client test
print("\n----- Test 2: Testing with OpenAI client -----", flush=True)
try:
    print("Importing OpenAI module...", flush=True)
    import openai
    from openai import OpenAI
    print(f"✓ Successfully imported openai (version: {openai.__version__})", flush=True)
    
    print("Configuring OpenAI client...", flush=True)
    client = OpenAI(
        base_url=LLAMA_SERVER_URL,
        api_key="not-needed",
        timeout=10.0
    )
    print("✓ Client configured", flush=True)
    
    # Test models endpoint
    print("Fetching models via OpenAI client...", flush=True)
    models = client.models.list()
    print(f"✓ Successfully retrieved {len(models.data)} models", flush=True)
    
    # Test simple chat completion
    print("\nTesting chat completion with a simple prompt...", flush=True)
    start_time = time.time()
    try:
        chat_response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0.7,
            max_tokens=20
        )
        elapsed = time.time() - start_time
        print(f"✓ Got response in {elapsed:.2f} seconds", flush=True)
        print(f"Response content: {chat_response.choices[0].message.content}", flush=True)
    except Exception as chat_error:
        print(f"✗ Chat completion error: {str(chat_error)}", flush=True)
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Error in Test 2: {str(e)}", flush=True)
    print("\nTraceback:", flush=True)
    traceback.print_exc()

print("\n===== TEST COMPLETED =====", flush=True)

