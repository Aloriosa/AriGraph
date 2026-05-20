"""
Analysis of Simformer results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    Analysis engine for Simformer results.
    """
    
    def __init__(self, results_path: str = "results"):
        """
        Initialize analysis engine.
        
        Args:
            results_path: Path to results directory
        """
        self.results_path = results_path
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """Load results from files."""
        for file in os.listdir(self.results_path):
            if file.endswith(".csv"):
                key = file.replace(".csv", "")
                self.results[key] = pd.read_csv(os.path.join(self.results_path, file))
    
    def plot_results(self, keys: List[str] = None):
        """
        Plot results.
        
        Args:
            keys: List of keys to plot
        """
        if keys is None:
            keys = list(self.results.keys())
        
        for key in keys:
            if key in self.results:
                plt.figure(figsize=(10, 5))
                plt.plot(self.results[key].iloc[:, 0], self.results[key].iloc[:, 1])
                plt.title(key)
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.grid(True)
                plt.savefig(os.path.join(self.results_path, f"{key}_plot.png"))
                plt.close()
    
    def calculate_coverage(self, key: str) -> float:
        """
        Calculate coverage.
        
        Args:
            key: Key to calculate coverage for
        Returns:
            coverage: Coverage value
        """
        if key in self.results:
            data = self.results[key]
            coverage = np.mean(data.iloc[:, 1])
            return coverage
        return 0.0
    
    def plot_coverage(self, keys: List[str] = None):
        """
        Plot coverage.
        
        Args:
            keys: List of keys to plot
        """
        if keys is None:
            keys = list(self.results.keys())
        
        coverage = []
        for key in keys:
            if key in self.results:
                coverage.append(self.calculate_coverage(key))
        
        plt.figure(figsize=(10, 5))
        plt.plot(keys, coverage)
        plt.title("Coverage")
        plt.xlabel("Key")
        plt.ylabel("Coverage")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, "coverage.png"))
        plt.close()
    
    def plot_all_results(self):
        """Plot all results."""
        self.plot_results()
        self.plot_coverage()
    
    def save_results(self, key: str, data: np.ndarray):
        """
        Save results.
        
        Args:
            key: Key to save results for
            data: Data to save
        """
        np.save(os.path.join(self.results_path, f"{key}.npy"), data)
    
    def save_results_csv(self, key: str, data: pd.DataFrame):
        """
        Save results as CSV.
        
        Args:
            key: Key to save results for
            data: Data to save
        """
        data.to_csv(os.path.join(self.results_path, f"{key}.csv"), index=False)
    
    def run_analysis(self, keys: List[str] = None):
        """
        Run analysis.
        
        Args:
            keys: List of keys to analyze
        """
        if keys is None:
            keys = list(self.results.keys())
        
        coverage_data = []
        for key in keys:
            if key in self.results:
            coverage = self.calculate_coverage(key)
            coverage_data.append({"key": key, "coverage": coverage})
        
        coverage_df = pd.DataFrame(coverage_data)
        self.save_results_csv("coverage_analysis", coverage_df)
        self.plot_coverage(keys)
    
    def plot_results_comparison(self, keys: List[str]):
        """
        Plot comparison of results.
        
        Args:
            keys: List of keys to compare
        """
        plt.figure(figsize=(10, 5))
        for key in keys:
            if key in self.results:
                plt.plot(self.results[key].iloc[:, 0], self.results[key].iloc[:, 1], label=key)
        plt.title("Comparison of Results")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, "results_comparison.png"))
        plt.close()
    
    def run_complete_analysis(self):
        """Run complete analysis."""
        self.load_results()
        self.run_analysis()
        self.plot_all_results()
        self.plot_results_comparison(list(self.results.keys()))
        self.save_results("analysis_complete", np.array([1.0]))