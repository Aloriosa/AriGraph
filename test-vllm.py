#!/usr/bin/env python3
"""
Test script for Qwen3-Coder-30B-A3B-Instruct via vLLM (OpenAI-compatible API)
Assumes SSH tunnel: local:1337 -> remote:1337
"""

from openai import OpenAI

BASE_URL = "http://localhost:4444/v1"
MODEL_NAME = "openai/gpt-oss-20b"

client = OpenAI(
    base_url=BASE_URL,
    api_key="none",  # vLLM doesn't require a real key by default
)


def check_health():
    """Check that the server is up and the model is loaded."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:4444/health", timeout=5) as r:
            print(f"[OK] Health endpoint: HTTP {r.status}")
    except Exception as e:
        print(f"[FAIL] Health endpoint unreachable: {e}")
        return False

    models = client.models.list()
    loaded = [m.id for m in models.data]
    print(f"[OK] Models available: {loaded}")
    if MODEL_NAME not in loaded:
        print(f"[WARN] Expected model '{MODEL_NAME}' not found in list above")
    return True


def test_basic_completion():
    """Simple single-turn chat completion."""
    print("\n--- Basic completion ---")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello and tell me what model you are."}],
        max_tokens=128,
        temperature=0.0,
    )
    print("Response:", response.choices[0].message.content)
    print("Finish reason:", response.choices[0].finish_reason)
    print("Usage:", response.usage)


def test_streaming():
    """Streaming completion."""
    print("\n--- Streaming completion ---")
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        max_tokens=64,
        temperature=0.0,
        stream=True,
    )
    print("Stream output: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def test_coding():
    """A real coding task — the primary use case for this model."""
    print("\n--- Coding task ---")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Write a Python function that returns the nth Fibonacci number using memoization. Include a brief docstring.",
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    print("=" * 50)
    print("vLLM connectivity test")
    print("=" * 50)

    if not check_health():
        print("\nServer not reachable. Check your SSH tunnel and try again.")
        exit(1)

    test_basic_completion()
    test_streaming()
    test_coding()

    print("\n[DONE] All tests passed.")