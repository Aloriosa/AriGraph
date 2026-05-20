import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Generation utilities
# ------------------------------------------------------------------
def generate_rationales(
    model,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = None,
):
    """
    Generate `n` chain‑of‑thought rationales for a given prompt.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    all_texts = []
    for _ in range(n):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # remove the prompt part
        text = text[len(prompt) :].strip()
        all_texts.append(text)
    return all_texts

# ------------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------------
ANSWER_PATTERNS = [
    r"The answer is\s+([^\s]+)",
    r"Answer is\s+([^\s]+)",
    r"Answer:\s+([^\s]+)",
    r"(\d+)",
    r"(\w+)",
]

def parse_answer(text: str):
    """
    Extract the final answer from a chain‑of‑thought text.
    Returns the string representation of the answer.
    """
    for pattern in ANSWER_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # fallback: last token
    return text.split()[-1].strip()

# ------------------------------------------------------------------
# Embedding utilities
# ------------------------------------------------------------------
class EmbeddingModel:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

# ------------------------------------------------------------------
# Weighting utilities
# ------------------------------------------------------------------
def centroid_proximity_weighting(embeddings, answers):
    """
    Compute CPW weights:
    - Compute centroid of embeddings.
    - Distance of each embedding to centroid.
    - Inverse of normalized distances -> weight.
    """
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    norm_dist = distances / np.sum(distances)
    # avoid division by zero
    norm_dist = np.where(norm_dist == 0, 1e-8, norm_dist)
    weights = 1.0 / norm_dist
    # accumulate weights for identical answers
    answer_to_weight = {}
    for ans, w in zip(answers, weights):
        answer_to_weight[ans] = answer_to_weight.get(ans, 0.0) + w
    best_answer = max(answer_to_weight.items(), key=lambda kv: kv[1])[0]
    return best_answer, answer_to_weight

def semantic_consensus_weighting(embeddings, answers):
    """
    Compute SCW weights:
    - Pairwise cosine similarity.
    - Sum similarities per answer.
    """
    # cosine similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sim_matrix, 0)  # ignore self
    scores = np.sum(sim_matrix, axis=1)
    answer_to_score = {}
    for ans, s in zip(answers, scores):
        answer_to_score[ans] = answer_to_score.get(ans, 0.0) + s
    best_answer = max(answer_to_score.items(), key=lambda kv: kv[1])[0]
    return best_answer, answer_to_score

# ------------------------------------------------------------------
# Self‑consistency baseline
# ------------------------------------------------------------------
def majority_vote(answers):
    """
    Return the most frequent answer. In case of tie, pick first encountered.
    """
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best_answer = max(counts.items(), key=lambda kv: kv[1])[0]
    return best_answer, counts