#!/usr/bin/env python3
"""
Generate plots for the reproduction results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, "r") as f:
        results = json.load(f)
    return results

def plot_lambada():
    """Plot LAMBADA results"""
    results = load_results("results/lambada_results.json")
    gamma = [r["gamma"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma, accuracy, "o-")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.title("LAMBADA Accuracy vs Gamma")
    plt.grid(True)
    plt.savefig("plots/lambada_performance.png")
    plt.close()

def plot_gsm8k():
    """Plot GSM8K results"""
    results = load_results("results/gsm8k_results.json")
    gamma = [r["gamma"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma, accuracy, "o-")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.title("GSM8K Accuracy vs Gamma")
    plt.grid(True)
    plt.savefig("plots/gsm8k_performance.png")
    plt.close()

def plot_humaneval():
    """Plot HumanEval results"""
    results = load_results("results/humaneval_results.json")
    gamma = [r["gamma"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    
    plt.figure(figsize=(10, 8))
    plt.plot(gamma, accuracy, "o-")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.title("HumanEval Accuracy vs Gamma")
    plt.grid(True)
    plt.savefig("plots/humaneval_performance.png")
    plt.close()

def plot_assistant():
    """Plot assistant results"""
    results = load_results("results/system_prompt_results.json")
    gamma = [r["gamma"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma, accuracy, "o-")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.title("Assistant Prompt Adherence Accuracy vs Gamma")
    plt.grid(True)
    plt.savefig("plots/assistant_comparisons.png")
    plt.close()

def main():
    """Main function"""
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate plots
    plot_lambada()
    plot_gsm8k()
    plot_humaneval()
    plot_assistant()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()