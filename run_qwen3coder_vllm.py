#!/usr/bin/env python3
"""
Single script to load and run 4-bit quantized Qwen3 Coder model via vLLM.
"""

from vllm import LLM, SamplingParams


def main():
    # Initialize vLLM with 4-bit quantized Qwen3 Coder
    print("Loading 4-bit quantized Qwen3 Coder model...")
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",  # 4-bit AWQ quantized version
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=1,
        max_model_len=4096,
    )
    print("✓ Model loaded successfully\n")
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    # Test prompts
    prompts = [
        "Write a Python function to calculate fibonacci numbers:",
        "Explain what is a decorator in Python:",
    ]
    
    print("Running inference...\n")
    print("="*70)
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Display results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i+1}: {prompt}")
        print(f"\nResponse:\n{generated_text}")
        print("="*70)
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
