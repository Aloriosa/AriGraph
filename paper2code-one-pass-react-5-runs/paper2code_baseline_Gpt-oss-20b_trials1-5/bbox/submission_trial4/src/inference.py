import os
import argparse
import torch

from .utils import load_config, read_csv, write_csv
from .blackbox_llm import BlackBoxLLM
from .adapter import Adapter
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_df = read_csv(os.path.join(cfg["data_dir"], "test.csv"))

    # Load tokenizer (shared between adapter and LLM)
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["adapter"]["encoder_name"])

    # Initialize LLM and adapter
    llm = BlackBoxLLM(cfg["llm_name"], device, cfg["generation"])
    adapter = Adapter(cfg["adapter"]["encoder_name"], cfg["adapter"]["dropout"]).to(device)

    adapter_path = os.path.join(cfg["output_dir"], "adapter.pt")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint not found at {adapter_path}")
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()

    # Prepare results
    results = []

    for q in tqdm(test_df["question"], desc="Inference"):
        # Generate candidates
        candidates = llm.generate(q, cfg["generation"]["num_candidates"])

        # Encode question
        q_enc = tokenizer(q, truncation=True, max_length=64, return_tensors="pt").to(device)

        # Score each candidate
        scores = []
        for c in candidates:
            c_enc = tokenizer(c, truncation=True, max_length=64, return_tensors="pt").to(device)
            score = adapter(q_enc["input_ids"], c_enc["input_ids"], torch.cat([q_enc["attention_mask"], c_enc["attention_mask"]], dim=1))
            scores.append(score.item())

        # Pick best
        best_idx = scores.index(max(scores))
        best_ans = candidates[best_idx]
        best_score = scores[best_idx]
        results.append({"question": q, "predicted_answer": best_ans, "score": best_score})

    # Write CSV
    out_path = os.path.join(cfg["output_dir"], "predictions.csv")
    write_csv(pd.DataFrame(results), out_path)
    print(f"Predictions written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)