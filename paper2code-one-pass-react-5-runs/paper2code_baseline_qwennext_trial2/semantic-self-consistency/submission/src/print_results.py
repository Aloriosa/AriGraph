import json
import sys

def main():
    """Print results summary"""
    try:
        with open("/home/submission/results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
        
        print("\n=== REPRODUCTION RESULTS SUMMARY ===")
        print(f"{'Dataset':<15} {'Baseline':<10} {'CPW':<10} {'SCW':<10} {'SCW > Baseline?'}")
        print("-" * 70)
        
        # Expected results from paper (GPT-3.5)
        expected = {
            "AQuA-RAT": {"baseline": 59.40, "cpw": 68.00, "scw": 65.40},
            "SVAMP": {"baseline": 79.80, "cpw": 81.00, "scw": 80.30},
            "StrategyQA": {"baseline": 66.81, "cpw": 65.21, "scw": 74.70}
        
        for dataset, result in results.items():
            baseline = result["accuracy"]  # Adjust based on your result structure
            cpw = result["accuracy"]  # Adjust
            scw = result["accuracy"]  # Adjust
            
            # For reproduction, we use the actual results
            print(f"{dataset:<15} {baseline:<10.2f} {cpw:<10.2f} {scw:<10.2f} {'✓' if scw > baseline else '✗'}")
        
        print("\nNote: Results may vary slightly due to API variability.")
        print("Expected: CPW and SCW should match or exceed baseline as per paper.")
        
    except FileNotFoundError:
        print("Results file not found. Run reproduce.sh first.")
        sys.exit(1)

if __name__ == "__main__":
    main()