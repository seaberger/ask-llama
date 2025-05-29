#!/usr/bin/env python3
"""
llama_query.py - Command-line interface for the local llama-server

This script allows you to send queries to a local llama-server directly from 
the command line and displays the formatted response in the terminal.

Usage:
    python llama_query.py "Your prompt here"
"""

import sys
import argparse
import time
import textwrap
from typing import Optional, Dict, List, Any
import json

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "dim": "\033[2m"
}

# Server configuration
SERVER_URL = "http://localhost:8080"
MODEL_ID = "/Users/seanbergman/Library/Caches/llama.cpp/ggml-org_Qwen2.5-Coder-3B-Q8_0-GGUF_qwen2.5-coder-3b-q8_0.gguf"

# Default parameters
DEFAULT_MAX_TOKENS = 500  # Reduced from 1000 to prevent repetition
DEFAULT_TEMPERATURE = 0.7
MIN_PARAGRAPH_LENGTH = 20  # Minimum characters to consider as a paragraph for deduplication

def color_text(text: str, color: str) -> str:
    """Apply color to text for terminal output."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def detect_and_remove_repetition(text: str) -> str:
    """Detect and remove repetitive paragraphs in the text."""
    # If text is very short, don't process
    if len(text) < MIN_PARAGRAPH_LENGTH * 2:
        return text
    
    # Split by paragraphs (empty lines)
    paragraphs = []
    current = []
    
    for line in text.split('\n'):
        if line.strip() == '':
            if current:
                paragraphs.append('\n'.join(current))
                current = []
        else:
            current.append(line)
    
    if current:
        paragraphs.append('\n'.join(current))
    
    # Skip processing if very few paragraphs
    if len(paragraphs) <= 1:
        return text
    
    # Detect and remove duplicates
    seen_paragraphs = set()
    unique_paragraphs = []
    
    for p in paragraphs:
        # Only consider paragraphs with substantial content
        if len(p.strip()) >= MIN_PARAGRAPH_LENGTH:
            if p not in seen_paragraphs:
                seen_paragraphs.add(p)
                unique_paragraphs.append(p)
        else:
            # Always keep short paragraphs
            unique_paragraphs.append(p)
    
    # If we removed lots of content, add a note
    if len(unique_paragraphs) < len(paragraphs) * 0.7:
        unique_paragraphs.append(color_text("\n(Note: Repetitive content was removed from the response)", "dim"))
    
    return '\n\n'.join(unique_paragraphs)

def trim_prompt_repetition(text: str, prompt: str) -> str:
    """Remove instances where the model repeats the prompt in the response."""
    if prompt in text:
        parts = text.split(prompt)
        # Keep the first part and join with newlines
        return parts[0].strip()
    return text

def format_code_blocks(text: str) -> str:
    """Format markdown code blocks with syntax highlighting."""
    lines = text.split('\n')
    in_code_block = False
    formatted_lines = []
    code_fence_pattern = '```'
    
    for line in lines:
        # Check for code fence markers with more flexibility
        if line.strip().startswith(code_fence_pattern):
            in_code_block = not in_code_block
            # Extract language if specified
            lang_part = line.strip()[3:].strip()
            if in_code_block and lang_part:
                formatted_lines.append(color_text(f"┌─ Code ({lang_part}) ", "yellow"))
                formatted_lines.append(color_text("│", "yellow"))
            elif in_code_block:
                formatted_lines.append(color_text("┌─ Code ", "yellow"))
                formatted_lines.append(color_text("│", "yellow"))
            else:
                formatted_lines.append(color_text("│", "yellow"))
                formatted_lines.append(color_text("└─────────", "yellow"))
        elif in_code_block:
            # Code within blocks is colored differently
            formatted_lines.append(color_text("│ ", "yellow") + color_text(line, "cyan"))
        else:
            # Regular text, apply wrapping
            formatted_lines.append(line)
            
    return '\n'.join(formatted_lines)

def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width, preserving code blocks."""
    lines = text.split('\n')
    wrapped_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            wrapped_lines.append(line)
        elif in_code_block or line.startswith('│ '):
            # Don't wrap code blocks
            wrapped_lines.append(line)
        else:
            # Wrap regular text
            if line.strip() == '':
                wrapped_lines.append(line)
            else:
                wrapped_parts = textwrap.wrap(line, width=width)
                wrapped_lines.extend(wrapped_parts if wrapped_parts else [''])
    
    return '\n'.join(wrapped_lines)

def create_spinner():
    """Create a simple spinner for showing progress."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    return frames

def query_llama_server(
    prompt: str, 
    system_message: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Dict[str, Any]:
    """Send a query to the llama-server and return the response."""
    try:
        from openai import OpenAI
    except ImportError:
        print(color_text("Error: OpenAI package not found. Installing...", "red"))
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
            from openai import OpenAI
        except Exception as e:
            print(color_text(f"Failed to install openai package: {e}", "red"))
            print(color_text("Please install it manually with: pip install openai", "yellow"))
            sys.exit(1)
    
    # Configure the client
    client = OpenAI(
        base_url=SERVER_URL,
        api_key="not-needed",  # Not required for local server
        timeout=60.0
    )
    
    # Prepare messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    # Start spinner
    spinner = create_spinner()
    spinner_idx = 0
    start_time = time.time()
    
    print(color_text("Sending query to llama-server... ", "blue"), end="", flush=True)
    
    try:
        # Show spinner while waiting
        response_future = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        # Get response
        response = response_future
        elapsed = time.time() - start_time
        
        print(color_text(f"✓ ({elapsed:.2f}s)", "green"))
        
        return {
            "content": response.choices[0].message.content,
            "elapsed": elapsed,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        print(color_text("✗", "red"))
        print(color_text(f"Error: {str(e)}", "red"))
        return {"error": str(e)}

def main():
    """Main function to parse arguments and run the query."""
    parser = argparse.ArgumentParser(description="Send queries to a local llama-server")
    parser.add_argument("prompt", nargs='+', help="The prompt to send to the model")
    parser.add_argument("--system", "-s", help="Optional system message to set context")
    parser.add_argument("--raw", "-r", action="store_true", help="Display raw output without formatting")
    parser.add_argument("--json", "-j", action="store_true", help="Output result as JSON")
    parser.add_argument("--width", "-w", type=int, default=80, help="Width for text wrapping (default: 80)")
    parser.add_argument("--tokens", "-t", type=int, default=DEFAULT_MAX_TOKENS, 
                      help=f"Maximum tokens in response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE,
                      help=f"Temperature for response generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--no-clean", action="store_true", 
                      help="Disable automatic cleaning of repetitive content")
    
    args = parser.parse_args()
    
    # Join prompt parts in case it was split by shell
    prompt = " ".join(args.prompt)
    
    if args.json:
        # JSON mode - just query and output as JSON
        result = query_llama_server(prompt, args.system)
        print(json.dumps(result, indent=2))
        return
    
    # Interactive mode with fancy output
    print(color_text("\n┌─ Query ", "magenta"))
    print(color_text("│ ", "magenta") + prompt)
    print(color_text("└─────────", "magenta"))
    
    result = query_llama_server(
        prompt, 
        args.system,
        temperature=args.temp,
        max_tokens=args.tokens
    )
    
    if "error" in result:
        print(color_text("\nError occurred:", "red"))
        print(result["error"])
        return
    
    print(color_text("\n┌─ Response ", "green"))
    
    if args.raw:
        # Raw output without formatting
        print(result["content"])
    else:
        # Get content and clean it if needed
        content = result["content"]
        
        # Apply cleaning steps unless disabled
        if not args.no_clean:
            # Remove repetitions of the prompt
            content = trim_prompt_repetition(content, prompt)
            # Remove repetitive paragraphs
            content = detect_and_remove_repetition(content)
            
        # Format and wrap the content
        formatted_content = format_code_blocks(content)
        wrapped_content = wrap_text(formatted_content, width=args.width)
        
        for line in wrapped_content.split('\n'):
            if not line.startswith('┌') and not line.startswith('└') and not line.startswith('│'):
                print(color_text("│ ", "green") + line)
            else:
                print(line)
    
    print(color_text("└─────────", "green"))
    
    # Print statistics
    if "tokens" in result:
        tokens = result["tokens"]
        stats = (
            f"Time: {result['elapsed']:.2f}s | "
            f"Tokens: {tokens['total']} ({tokens['prompt']} prompt, {tokens['completion']} completion)"
        )
        print(color_text(stats, "dim"))
    
    print()  # Add final newline

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nOperation cancelled by user", "yellow"))
        sys.exit(0)
    except Exception as e:
        print(color_text(f"\nUnexpected error: {e}", "red"))
        sys.exit(1)

