import asyncio
import httpx
import time
import random
import string

# --- Configuration ---
API_URL = "http://localhost:10240/v1/embeddings"
NUM_REQUESTS = 50
STRING_LENGTH = 1000
STRINGS_PER_BATCH = 16
RANDOM_SEED = 42
MODEL_NAME = "mlx-community/multilingual-e5-large" # Use a placeholder model name as required by the OpenAI API spec

def generate_random_strings(count: int, length: int, batch_size: int, seed: int) -> list[list[str]]:
    """Generates a list of reproducible random strings."""
    random.seed(seed)
    print(f"Generating {count} reproducible strings (Length: {length}) with seed {seed}...")
    
    chars = string.ascii_letters + string.digits + ' '
    strings = []
    for _ in range(count):
        # Generate string and prepend a unique identifier for tracking if needed
        strings.append([''.join(random.choice(chars) for _ in range(length)) for i in range(batch_size)])
    
    print("Generation complete.")
    return strings

async def call_embedding_api(client: httpx.AsyncClient, data_strings: list[str], index: int) -> dict:
    """
    Sends a single asynchronous embedding request.

    Args:
        client: The shared httpx.AsyncClient instance.
        data_strings: The strings to be embedded.
        index: The index of the request for logging purposes.
    
    Returns:
        A dictionary containing the request result.
    """
    payload = {
        "input": data_strings,
        "model": MODEL_NAME
    }
    
    try:
        # Start timer for this individual request
        start_time = time.monotonic()
        
        # Send POST request
        response = await client.post(
            API_URL, 
            json=payload, 
            timeout=120.0 # Set a reasonable timeout
        )
        
        # Calculate individual request time
        duration = time.monotonic() - start_time
        
        # Check for successful response
        response.raise_for_status()
        
        # Optional: Print success and timing for this request
        print(f"Request {index:02d} completed successfully in {duration:.4f}s. Status: {response.status_code}")
        
        return {
            "index": index,
            "status": "success",
            "time_s": duration
        }
    
    except httpx.HTTPStatusError as e:
        # Handle non-200 responses
        error_msg = f"HTTP Error {e.response.status_code} for request {index:02d}: {e.response.text.strip()}"
        print(error_msg)
        return {"index": index, "status": "http_error", "message": error_msg}
    
    except httpx.RequestError as e:
        # Handle connection errors (e.g., endpoint not running)
        error_msg = f"Connection Error for request {index:02d}: {e}"
        print(error_msg)
        return {"index": index, "status": "connection_error", "message": error_msg}


async def main():
    """Main function to orchestrate the parallel requests and measure time."""
    
    # 1. Generate the reproducible inputs
    input_strings = generate_random_strings(NUM_REQUESTS, STRING_LENGTH, STRINGS_PER_BATCH, RANDOM_SEED)
    
    # 2. Use a shared AsyncClient for better connection pooling and performance
    # Setting limits=None removes the default concurrency limit of 100
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=20)) as client:
        
        # 3. Create a list of tasks
        tasks = []
        for i, s in enumerate(input_strings):
            task = call_embedding_api(client, s, i + 1)
            tasks.append(task)
        
        print("\n--- Starting Parallel Requests ---")
        start_global_time = time.monotonic()
        
        # 4. Run all tasks concurrently and wait for them to finish
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # 5. Measure the total time
        end_global_time = time.monotonic()
        total_wall_time = end_global_time - start_global_time

    print("\n--- Benchmarking Complete ---")
    print(f"Total Requests Sent: {NUM_REQUESTS}")
    print(f"Total Wall Time (Time for all to complete): {total_wall_time:.4f} seconds")
    
    # Optional: Report successful vs failed requests
    successful_requests = sum(1 for r in results if r and r.get('status') == 'success')
    print(f"Successful Requests: {successful_requests}/{NUM_REQUESTS}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
