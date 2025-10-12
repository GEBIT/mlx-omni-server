from openai import OpenAI

client = OpenAI(base_url="http://localhost:10240/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in completion:
    if hasattr(chunk.choices[0].delta, "reasoning"):
        print(chunk.choices[0].delta.reasoning.lower(), end="", flush=True)
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content.upper(), end="", flush=True)

print("\n---")
