#!/bin/bash
# Script to start llama-server with minimal parameters

SERVER_PATH="/opt/homebrew/Cellar/llama.cpp/5370/bin/llama-server"
MODEL_PATH="/Users/seanbergman/Library/Caches/llama.cpp/Qwen3-4B-Q8_0.gguf"
LOG_FILE="/tmp/llama-server.log"

# Kill any existing server
pkill -f "llama-server" >/dev/null 2>&1
sleep 2

# Start server with minimal parameters
echo "Starting llama-server..."
"$SERVER_PATH" --model "$MODEL_PATH" --jinja --n-gpu-layers 1 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Server started with PID $PID"

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 5

# Check server health
if curl -s "http://localhost:8080/health" > /dev/null; then
    echo "Server is healthy"
    curl -s "http://localhost:8080/health"
else
    echo "Server health check failed"
    echo "Server log:"
    tail -20 "$LOG_FILE"
    exit 1
fi
