#!/usr/bin/env python3
"""
Simple script to test LLM availability and connectivity.
Tests the API endpoint used in the paper reproduction pipeline.
"""

import sys
import argparse
from openai import OpenAI, DefaultHttpxClient


def test_llm_availability():
    """Test if the LLM endpoint is available and responding."""
    base_url = "https://inference.airi.net:46783/v1"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3Njk0NDQ0MjQsImV4cCI6MTc3MDA0OTIyNH0.4eyt0_zsvfDY4HmwDsc4eS0p0mFDftFQL7u_DhRJqt4"

    print("="*70)
    print("LLM AVAILABILITY TEST")
    print("="*70)
    print(f"Base URL: {base_url}")
    
    try:
        print("Initializing OpenAI client...")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(verify=False)
        )
        print("✓ Client initialized successfully")
        
        models = client.models.list()
        model = None
        for i in models.data:
            if i.id == 'Qwen/Qwen3-Coder-30B-A3B-Instruct':
                model = i.id
                break
        if model is None:
            raise ValueError('model not found')
            
        print(f"Sending test request to {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello! Please respond with 'OK' if you can read this."}],
            temperature=0.1,
            max_tokens=50
        )
        print("✓ Received response from LLM")
        print("")
        
        # Extract response
        message = response.choices[0].message.content
        print("Response content:")
        print(f"  '{message}'")
        print("")
        
        # Check usage info if available
        if hasattr(response, 'usage') and response.usage:
            print("Usage statistics:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
            print("")
        
        print("="*70)
        print("✓ SUCCESS: LLM is available and responding correctly!")
        print("="*70)
        return True
        
    except Exception as e:
        print("")
        print("="*70)
        print("✗ ERROR: Failed to connect to LLM")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("")
        print("Troubleshooting tips:")
        print("  1. Check if the API key is valid and not expired")
        print("  2. Verify the base URL is correct and accessible")
        print("  3. Ensure you have network connectivity")
        print("  4. Check if the model name is correct")
        return False


def main():
    
    
    success = test_llm_availability()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
