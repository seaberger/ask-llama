#!/usr/bin/env python3

import sys
import http.client
import json
import traceback

# Function to write to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

# Start
eprint("\n=== BASIC LLAMA-SERVER CONNECTION TEST ===")
eprint(f"Python version: {sys.version}")

try:
    # Make a direct HTTP connection
    eprint("Connecting to localhost port 8080...")
    conn = http.client.HTTPConnection("localhost", 8080, timeout=5)
    
    # Send GET request to models endpoint
    eprint("Sending GET request to /v1/models...")
    conn.request("GET", "/v1/models")
    
    # Get the response
    eprint("Waiting for response...")
    response = conn.getresponse()
    
    # Output status code
    status = response.status
    reason = response.reason
    eprint(f"Response status: {status} {reason}")
    
    # Read response data
    eprint("Reading response data...")
    data = response.read().decode('utf-8')
    
    # Output headers
    eprint("\nResponse headers:")
    for header in response.getheaders():
        eprint(f"  {header[0]}: {header[1]}")
    
    # Output data preview
    preview_length = 500
    data_preview = data[:preview_length] + "..." if len(data) > preview_length else data
    eprint("\nResponse data preview:")
    eprint(data_preview)
    
    # Try to parse as JSON
    try:
        eprint("\nParsing JSON response...")
        json_data = json.loads(data)
        
        if 'data' in json_data and isinstance(json_data['data'], list):
            models = json_data['data']
            eprint(f"Found {len(models)} models:")
            
            for model in models:
                model_id = model.get('id', 'unknown')
                eprint(f"  - {model_id}")
        else:
            eprint("JSON response doesn't contain expected 'data' list")
    
    except json.JSONDecodeError as je:
        eprint(f"Failed to parse JSON: {je}")
    
    # Close connection
    conn.close()
    eprint("Connection closed")

except Exception as e:
    eprint(f"\nERROR: {str(e)}")
    eprint("\nTraceback:")
    traceback.print_exc(file=sys.stderr)

eprint("\n=== TEST COMPLETED ===")

