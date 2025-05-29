#!/usr/bin/env python3

import sys
import os
import traceback
import time

# Use absolute path in home directory
log_path = os.path.join(os.path.expanduser("~"), "llama_test_output.log")
sys.stderr.write(f"Writing logs to: {log_path}\n")
sys.stderr.flush()

# Create log file
try:
    log_file = open(log_path, "w")
    sys.stderr.write("Log file opened successfully\n")
except Exception as e:
    sys.stderr.write(f"ERROR: Could not open log file: {str(e)}\n")
    sys.stderr.flush()
    raise

def log(message, error=False):
    """Write message to log file and stderr if it's an error"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    
    # Always write to log file
    log_file.write(formatted_msg + "\n")
    log_file.flush()
    
    # Write to stderr for errors or stdout for normal messages
    if error:
        sys.stderr.write(formatted_msg + "\n")
        sys.stderr.flush()
    else:
        try:
            print(formatted_msg, flush=True)
        except:
            sys.stderr.write(formatted_msg + "\n")
            sys.stderr.flush()

log("=== STARTING MINIMAL LLAMA-SERVER TEST ===")
log(f"Python version: {sys.version}")
log(f"Script location: {__file__}")
log(f"Current working directory: {os.getcwd()}")
log(f"Log file path: {log_path}")
log(f"System info: {sys.platform}")

try:
    log("Importing requests library...")
    import requests
    log("Successfully imported requests")
    
    server_url = "http://localhost:8080"
    log(f"Testing connection to {server_url}/v1/models")
    
    response = requests.get(f"{server_url}/v1/models", timeout=5)
    log(f"Response status code: {response.status_code}")
    log(f"Response headers: {dict(response.headers)}")
    
    # Log the content but also save it to a separate file for easier viewing
    content = response.text
    content_path = os.path.join(os.path.expanduser("~"), "llama_server_response.json")
    log(f"Saving full response to: {content_path}")
    
    with open(content_path, "w") as f:
        f.write(content)
        f.flush()
    
    # Also log a preview of the content
    content_preview = content[:500] + "..." if len(content) > 500 else content
    log(f"Response content preview: {content_preview}")
    
    if response.status_code == 200:
        log("Connection successful!")
        data = response.json()
        models = data.get('data', [])
        log(f"Found {len(models)} models:")
        for model in models:
            log(f"  - {model.get('id')}")
    else:
        log(f"Error connecting to server. Status code: {response.status_code}")

except Exception as e:
    error_msg = f"ERROR: {str(e)}"
    log(error_msg, error=True)
    log("Traceback:", error=True)
    
    # Log traceback to both log file and stderr
    traceback_text = traceback.format_exc()
    log_file.write(traceback_text + "\n")
    log_file.flush()
    sys.stderr.write(traceback_text + "\n")
    sys.stderr.flush()

log("=== TEST COMPLETED ===")
log(f"Log file written to: {log_path}")
log_file.close()

# Print final instruction to stderr to ensure visibility
sys.stderr.write(f"\nTest completed. View detailed logs at: {log_path}\n")
sys.stderr.flush()

