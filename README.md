# ask-llama CLI Tool

A command-line interface for interacting with the local llama.cpp server and Qwen3-4B model.

## Overview

The `ask-llama` tool provides a streamlined interface for communicating with a local LLM server powered by llama.cpp. It allows you to send queries to the Qwen3-4B model and view the responses with various formatting options, including the ability to see the model's thinking process.

## Files

- `query-llama.py`: Main Python script that interfaces with the llama-server
- `start-llama-server.sh`: Script to start and manage the llama-server instance
- `.zshrc` integration: Shell function that provides the `ask-llama` command

## Installation

1. Ensure dependencies are installed:
   ```bash
   pip install requests
   ```

2. Make sure the scripts are executable:
   ```bash
   chmod +x query-llama.py start-llama-server.sh
   ```

3. Update your .zshrc to include the ask-llama function (already done)

## Usage

### Basic Usage

```bash
ask-llama "What is the capital of France?"
```

### With Thinking Process

```bash
ask-llama --think "Explain quantum computing"
```

### With Custom System Prompt

```bash
ask-llama --system "You are a helpful coding assistant. Provide code examples." "How do I implement a binary search in Python?"
```

### Full Command Options

```bash
ask-llama [options] "Your question here"
```

## Command-line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--system` | `-s` | Custom system prompt | "You are a helpful AI assistant. Analyze thoroughly, then give a direct answer." |
| `--temp` | `-t` | Temperature for response randomness (0.0-1.0) | 0.7 |
| `--tokens` | `-n` | Maximum number of tokens to generate | Full context length (32k) |
| `--width` | `-w` | Width for text wrapping | 80 |
| `--think` | | Show the model's thinking process | Off |
| `--no-color` | | Disable colored output | Off |
| `--counts` | `-c` | Show token counts and timing | Off |

## Examples

### Basic Question
```bash
ask-llama "What causes tides on Earth?"
```

### Showing Thinking Process
```bash
ask-llama --think "What are the implications of Moore's Law slowing down?"
```

### Adjusting Response Randomness
```bash
ask-llama --temp 0.2 "Give me ideas for a science fiction story"
```

### Creative Writing
```bash
ask-llama --system "You are a creative writing assistant" --temp 0.8 "Write a short poem about autumn"
```

### Code Generation with Thinking
```bash
ask-llama --think --system "You are a Python expert" "Write a function to find prime numbers up to n"
```

### Token Usage Analysis
```bash
ask-llama --counts "Explain the Big Bang theory in simple terms"
```

## Server Management

The server starts automatically when needed, but you can also manage it manually:

- By default, the server is launched with the `--jinja` flag. This enables OpenAI-style function calling (tool use) using the Jinja engine.
- Check if server is running: `pgrep -f "llama-server"`
- Manually start server: `~/start-llama-server.sh`
- Stop server: `pkill -f "llama-server"`

## Troubleshooting

- If you encounter connection errors, check if the server is running
- For slow responses, consider using a smaller value for `--tokens`
- If the output appears truncated, increase the `--tokens` value

## Notes

- The default model is Qwen3-4B-Q8_0, located at `/Users/seanbergman/Library/Caches/llama.cpp/Qwen3-4B-Q8_0.gguf`
- The server runs locally on port 8080
- Thinking mode is especially useful for complex reasoning tasks
- The server always starts with the `--jinja` flag, providing built-in support for advanced function and tool calling via Jinja templating.
