#!/usr/bin/env python3
import requests
import argparse
import textwrap
import sys
import json
import re
import time

# Model constants
MAX_CONTEXT_LENGTH = 32768  # Qwen3 model's max context length

# ANSI color codes for terminal output
COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'italic': '\033[3m',
    'underline': '\033[4m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'grey': '\033[90m',
    'think': '\033[38;5;246m',  # Grey color for think blocks
    'think_border': '\033[38;5;240m',  # Darker grey for think block borders
}

def query_server(prompt, system=None, temperature=0.7, max_tokens=None, show_thinking=False):
    """Send a query to the llama-server."""
    # Prepare system message
    if system:
        system_msg = system
    else:
        system_msg = "You are a helpful AI assistant."
        
    if show_thinking:
        system_msg += "\nFirst analyze step by step in <think> tags, then give a direct answer."
        prompt = prompt + " /think"  # Add think tag for Qwen model
        
    # Format prompt with chat template
    formatted_prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Create payload
    payload = {
        "prompt": formatted_prompt,
        "temperature": temperature,
        "top_k": 40,
        "top_p": 0.9,
        "stream": False,
        "stop": ["</s>", "<|im_end|>"]
    }
    
    # Only set n_predict if a specific token limit was requested
    if max_tokens is not None:
        payload["n_predict"] = max_tokens
    
    try:
        # Send request
        response = requests.post(
            "http://localhost:8080/completion",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        
        # Return both the content and the full response (for token counting)
        return {
            "content": response_json.get("content", ""),
            "full_response": response_json
        }
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def format_thinking_block(text, width):
    """Format a thinking block with nice borders."""
    lines = text.strip().split('\n')
    
    # Create border
    border = "─" * 40
    
    # Format header
    result = [
        f"\033[1m\033[38;5;246m┌{border}┐\033[0m",
        f"\033[38;5;246m│\033[1m THINKING PROCESS {' ' * 23}\033[38;5;246m│\033[0m"
    ]
    
    # Format content with wrapping
    for line in lines:
        if line.strip():
            wrapped = textwrap.wrap(line, width=width-4)
            for wrapped_line in wrapped:
                padding = ' ' * (width - 4 - len(wrapped_line))
                result.append(f"\033[38;5;246m│\033[0m \033[38;5;246m{wrapped_line}{padding}\033[0m \033[38;5;246m│\033[0m")
        else:
            result.append(f"\033[38;5;246m│{' ' * (width-2)}│\033[0m")
    
    # Add footer
    result.append(f"\033[38;5;246m└{border}┘\033[0m")
    
    return '\n'.join(result)

def count_tokens_in_text(text):
    """Estimate token count in text (rough approximation)."""
    if not text:
        return 0
        
    # This is a very rough approximation of token count
    # For a more accurate count, you'd need a tokenizer
    # Average English word is ~1.3 tokens
    words = text.split()
    return int(len(words) * 1.3)


def format_response(text, width=80, show_thinking=False, use_color=True, tokens=None, 
                   show_counts=False, full_response=None):
    """Format the response with proper wrapping and thinking blocks."""
    # Remove chat markers
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    
    # Handle thinking blocks
    think_pattern = r'<think>(.*?)</think>'
    thinking_blocks = re.findall(think_pattern, text, re.DOTALL)
    
    # Extract thinking content for token counting
    thinking_content = "\n".join(thinking_blocks) if thinking_blocks else ""
    
    if thinking_blocks and show_thinking:
        for block in thinking_blocks:
            formatted_block = format_thinking_block(block, width) if use_color else block
            pattern = re.escape(f"<think>{block}</think>")
            text = re.sub(pattern, formatted_block, text, flags=re.DOTALL)
    else:
        # Remove thinking blocks if not showing
        text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    
    # Only check for truncation if a token limit was specified or if there are
    # other signs of truncation in the text
    truncation_indicators = ["...", "…", "to be continued", "cont", "continues"]
    is_truncated = any(text.rstrip().endswith(indicator) for indicator in truncation_indicators)
    
    # Also check token limit if specified
    if tokens is not None and len(text) >= tokens - 20:  # Close to limit
        is_truncated = True
        
    if is_truncated:
        text += "\n\n\033[33m[Note: Response may be truncated]\033[0m" if use_color else "\n\n[Note: Response may be truncated]"
    
    # Clean up any remaining tags
    text = text.replace("<think>", "").replace("</think>", "")
    
    # Clean up extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Add token counts if requested
    if show_counts:
        try:
            # Create a clean version of text without the formatted thinking blocks
            clean_text = text
            for block in thinking_blocks:
                if show_thinking:
                    formatted_block = format_thinking_block(block, width) if use_color else f"\n[Thinking Process]\n{block}\n"
                    clean_text = clean_text.replace(formatted_block, "")
                else:
                    # If thinking is not shown, the text is already clean
                    pass
                    
            # Remove any remaining tags or formatting
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
            answer_content = clean_text.strip()
            
            # Get actual token counts from the server response if available
            prompt_tokens = 0
            completion_tokens = 0
            thinking_tokens = 0
            answer_tokens = 0
            total_tokens = 0
            generation_time = 0
            tokens_per_second = 0
            
            if full_response:
                # Extract token counts from the server response
                prompt_tokens = full_response.get("tokens_evaluated", 0)
                completion_tokens = full_response.get("tokens_predicted", 0)
                
                # Estimate thinking vs answer tokens
                if thinking_blocks and show_thinking:
                    # Estimate the proportion of tokens used for thinking vs answer
                    thinking_text = "\n".join(thinking_blocks)
                    thinking_char_ratio = len(thinking_text) / (len(thinking_text) + len(answer_content) + 1)
                    thinking_tokens = int(completion_tokens * thinking_char_ratio)
                    answer_tokens = completion_tokens - thinking_tokens
                else:
                    answer_tokens = completion_tokens
                    thinking_tokens = 0
                    
                total_tokens = prompt_tokens + completion_tokens
                
                # Get timing information if available
                if "timings" in full_response:
                    timings = full_response["timings"]
                    prompt_time = timings.get("prompt_ms", 0) / 1000  # Convert to seconds
                    generation_time = timings.get("predicted_ms", 0) / 1000  # Convert to seconds
                    tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            else:
                # Estimate token counts if server data not available
                prompt_tokens = count_tokens_in_text(full_response.get("prompt", ""))
                thinking_tokens = count_tokens_in_text("\n".join(thinking_blocks)) if thinking_blocks else 0
                answer_tokens = count_tokens_in_text(answer_content)
                total_tokens = prompt_tokens + thinking_tokens + answer_tokens
        
            # Format token counts
            if use_color:
                counts_text = f"\n\n{COLORS['cyan']}══════ Token Counts ══════{COLORS['reset']}\n"
                counts_text += f"Prompt tokens: {COLORS['bold']}{prompt_tokens}{COLORS['reset']}\n"
                if thinking_blocks and show_thinking:
                    counts_text += f"Thinking tokens: {COLORS['bold']}{thinking_tokens}{COLORS['reset']}\n"
                counts_text += f"Answer tokens: {COLORS['bold']}{answer_tokens}{COLORS['reset']}\n"
                counts_text += f"Total tokens: {COLORS['bold']}{total_tokens}{COLORS['reset']}\n"
                
                if generation_time > 0:
                    counts_text += f"Generation time: {COLORS['bold']}{generation_time:.2f}{COLORS['reset']} seconds\n"
                    counts_text += f"Tokens per second: {COLORS['bold']}{tokens_per_second:.2f}{COLORS['reset']}\n"
            else:
                counts_text = "\n\n══════ Token Counts ══════\n"
                counts_text += f"Prompt tokens: {prompt_tokens}\n"
                if thinking_blocks and show_thinking:
                    counts_text += f"Thinking tokens: {thinking_tokens}\n"
                counts_text += f"Answer tokens: {answer_tokens}\n"
                counts_text += f"Total tokens: {total_tokens}\n"
                
                if generation_time > 0:
                    counts_text += f"Generation time: {generation_time:.2f} seconds\n"
                    counts_text += f"Tokens per second: {tokens_per_second:.2f}\n"
            
            text += counts_text
        except Exception as e:
            # If token counting fails, add a simple message
            if use_color:
                text += f"\n\n{COLORS['yellow']}[Note: Token counting failed: {str(e)}]{COLORS['reset']}"
            else:
                text += f"\n\n[Note: Token counting failed: {str(e)}]"
    
    return text

def main():
    parser = argparse.ArgumentParser(description='Query the llama-server')
    parser.add_argument('prompt', help='The prompt to send')
    parser.add_argument('--system', '-s', help='System message')
    parser.add_argument('--temp', '-t', type=float, default=0.7, help='Temperature')
    parser.add_argument('--tokens', '-n', type=int, help='Max tokens (default: use model\'s full context)')
    parser.add_argument('--width', '-w', type=int, default=80, help='Output width')
    parser.add_argument('--think', action='store_true', help='Show thinking process')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--counts', '-c', action='store_true', help='Show token counts and timing information')
    args = parser.parse_args()
    
    # Set token defaults appropriately if specified
    token_limit = None
    if args.tokens is not None:
        token_limit = args.tokens
        # Increase tokens if using think mode with a low token count
        if args.think and token_limit < 512:
            token_limit = 512
    
    # Start timing for query execution
    start_time = time.time()
    
    # Get response from server
    response_data = query_server(
        args.prompt,
        args.system,
        args.temp,
        token_limit,
        args.think
    )
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Check for empty content and handle
    content = response_data.get("content", "")
    if not content.strip():
        print("Error: Received empty response from server.")
        sys.exit(1)
        
    # Format response with token counts if requested
    formatted = format_response(
        content, 
        args.width, 
        args.think,
        not args.no_color,
        token_limit,
        args.counts,
        response_data["full_response"] if args.counts else None
    )
    
    print(formatted)

if __name__ == "__main__":
    main()
