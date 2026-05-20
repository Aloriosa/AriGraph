#!/usr/bin/env python3
"""
Prepare datasets for APT reproduction
"""
import os
from datasets import load_dataset
import torch

def prepare_sst2():
    """Prepare SST-2 dataset"""
    print("Downloading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Save dataset to disk
    os.makedirs("data/sst2", exist_ok=True)
    dataset.save_to_disk("data/sst2")
    print(f"SST-2 dataset saved with {len(dataset['train'])} training samples")

def prepare_cnn_dailymail():
    """Prepare CNN/DailyMail dataset"""
    print("Downloading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Save dataset to disk
    os.makedirs("data/cnn_dailymail", exist_ok=True)
    dataset.save_to_disk("data/cnn_dailymail")
    print(f"CNN/DailyMail dataset saved with {len(dataset['train'])} training samples")

def prepare_alpaca():
    """Prepare Alpaca dataset (dummy version for reproduction)"""
    print("Creating dummy Alpaca dataset...")
    
    # Create a small dummy dataset for demonstration
    # In a real reproduction, this would be the actual Alpaca dataset
    dummy_data = {
        "instruction": [
            "Explain the concept of gravity",
            "What is the capital of France?",
            "Write a poem about春天",
            "Summarize the main points of the theory of relativity",
            "Translate 'Hello, how are you?' to Spanish"
        ],
        "input": ["", "", "", "", ""],
        "output": [
            "Gravity is a fundamental force that attracts objects with mass towards each other.",
            "The capital of France is Paris.",
            "春天是万物复苏的季节，花朵绽放，鸟儿歌唱，阳光温暖。",
            "The theory of relativity, developed by Albert Einstein, describes how space and time are interwoven into a single continuum known as spacetime.",
            "Hola, ¿cómo estás?"
        ]
    }
    
    os.makedirs("data/alpaca", exist_ok=True)
    torch.save(dummy_data, "data/alpaca/dummy_dataset.pt")
    print(f"Dummy Alpaca dataset created with {len(dummy_data['instruction'])} samples")

if __name__ == "__main__":
    prepare_sst2()
    prepare_cnn_dailymail()
    prepare_alpaca()
    print("All datasets prepared successfully!")