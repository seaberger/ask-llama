#!/usr/bin/env python3
"""
chat_test.py - Simple demonstration of using OpenAI client with local llama-server

This script shows how to configure the OpenAI client to interact with a locally
running llama-server for chat completions.
"""

import sys
import time
import traceback

# Set server constants
LLAMA_SERVER_URL = "http://localhost:8080"
MODEL_ID = "/Users/seanbergman/Library/Caches/llama.cpp/ggml-org_Qwen2.5-Coder-3B-Q8_0-GGUF_qwen2.5-coder-3b-q8_0.gguf"

def print_section(title):
    """Print a section title with formatting"""
    print(f"\n{'=' * 5} {title} {'=' * 5}", file=sys.stderr)

print_section("CHAT TEST WITH LOCAL LLAMA-SERVER")
print(f"Python version: {sys.version}", file=sys.stderr)

try:
    print("Importing OpenAI module...", file=sys.stderr)
    import openai
    from openai import OpenAI
    print(f"OpenAI version: {openai.__version__}", file=sys.stderr)
    
    # Configure the OpenAI client to use the local server
    print_section("CONFIGURING CLIENT")
    print(f"Setting up client with base URL: {LLAMA_SERVER_URL}", file=sys.stderr)
    
    client = OpenAI(
        base_url=LLAMA_SERVER_URL,  # Local llama-server URL
        api_key="not-needed",       # No API key needed for local server
        timeout=30.0                # Longer timeout for generation
    )
    print("Client configured successfully", file=sys.stderr)
    
    # Verify models are available
    print_section("CHECKING AVAILABLE MODELS")
    try:
        models = client.models.list()
        print(f"Available models: {len(models.data)}", file=sys.stderr)
        for model in models.data:
            print(f"  - {model.id}", file=sys.stderr)
    except Exception as model_error:
        print(f"Error listing models: {model_error}", file=sys.stderr)
        # Continue anyway as we already know our model ID
    
    # Set up chat parameters
    print_section("CHAT COMPLETION DEMO")
    
    # Sample messages for a chat conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that provides concise answers."},
        {"role": "user", "content": "Write a short Python function to check if a number is prime."}
    ]
    
    print("Sending chat completion request with the following conversation:", file=sys.stderr)
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}", file=sys.stderr)
    
    # Record start time to measure response time
    print("\nWaiting for response...", file=sys.stderr)
    start_time = time.time()
    
    # Send the chat completion request
    chat_completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.7,      # Controls randomness
        max_tokens=300,       # Limit response length
        stream=False          # Get complete response at once
    )
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Extract and display the response
    response_content = chat_completion.choices[0].message.content
    
    print(f"\nResponse received in {elapsed:.2f} seconds:", file=sys.stderr)
    print_section("MODEL RESPONSE")
    print(response_content)
    
    # Print token usage information
    print_section("USAGE STATISTICS")
    print(f"Prompt tokens: {chat_completion.usage.prompt_tokens}", file=sys.stderr)
    print(f"Completion tokens: {chat_completion.usage.completion_tokens}", file=sys.stderr)
    print(f"Total tokens: {chat_completion.usage.total_tokens}", file=sys.stderr)
    
except Exception as e:
    print_section("ERROR")
    print(f"An error occurred: {str(e)}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print_section("COMPLETED SUCCESSFULLY")

