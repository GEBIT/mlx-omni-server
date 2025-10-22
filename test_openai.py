from argparse import ArgumentParser
from openai import OpenAI
import sys

# Define the mapping for user-friendly model names to actual model IDs
MODEL_MAPPING = {
    "gpt-oss:20b": "mlx-community/gpt-oss-20b-MXFP4-Q4",
    "gpt-oss:120b": "mlx-community/gpt-oss-120b-MXFP4-Q4",
    "gemma3:27b": "mlx-community/gemma-3-27b-it-4bit",
    "gemma3:4b": "mlx-community/gemma-3-text-4b-it-4bit",
    "gemma3:1b": "mlx-community/gemma-3-4b-it-4bit-DWQ",
    "qwen2.5-coder:32b": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    "deepseek-r1:70b": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
    "qwen3:30b": "Qwen/Qwen3-30B-A3B-MLX-4bit"
}

def get_model_id(model_alias):
    """Translates the model alias to its corresponding model ID."""
    return MODEL_MAPPING.get(model_alias)

def main():
    """
    Parses arguments and runs the OpenAI chat completion.
    """
    parser = ArgumentParser(description="Run an OpenAI-compatible chat completion.")
    
    # Get available model aliases for the help message
    model_choices = ", ".join(MODEL_MAPPING.keys())
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=MODEL_MAPPING.keys(),
        help=f"The model alias to use. Choices: {model_choices}"
    )
    
    args = parser.parse_args()
    
    # Get the actual model ID from the mapping
    model_id = get_model_id(args.model)
    
    if not model_id:
        print(f"Error: Model alias '{args.model}' is not mapped to an ID.", file=sys.stderr)
        sys.exit(1)

    # Initialize the client
    client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

    print(f"Using model: {model_id}\n---")

    # Run the chat completion
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        # Process the stream
        for chunk in completion:
            # Check for 'reasoning' (common in MLX/local models)
            if hasattr(chunk.choices[0].delta, "reasoning") and chunk.choices[0].delta.reasoning is not None:
                print(chunk.choices[0].delta.reasoning.lower(), end="", flush=True)
            # Check for 'content'
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content.upper(), end="", flush=True)

        print("\n---")
        
    except Exception as e:
        print(f"\nError during API call: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()