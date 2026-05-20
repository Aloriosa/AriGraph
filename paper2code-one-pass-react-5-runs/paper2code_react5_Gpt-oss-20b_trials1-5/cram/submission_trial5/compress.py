#!/usr/bin/env python3
"""
Memory Vector Compression Demo – Reproduction of core metrics from
"Cramming 1568 Tokens into a Single Vector and Back Again".

The script implements the per‑sample optimisation of trainable memory vectors
(`[mem]`) for a frozen language model and computes the following metrics
for each text, length and number of memory vectors `k`:

* Decoding Capacity – the largest token length `L` for which the
  teacher‑forcing accuracy with `[mem]` is ≥ `--threshold`.
* Teacher‑forcing accuracy.
* Free‑generation accuracy.
* Token Gain – difference in the number of correctly predicted tokens
  with and without `[mem]`.
* Information Gain – reduction of cross‑entropy (bits per token) when
  using `[mem]`.

The script can run on natural text (provided via `--text-file`) or on
random word sequences (`--random`).  Results are written to CSV files
in the `results/` directory.  A separate `decoding_capacity.csv`
file summarises the maximum lengths that satisfy the accuracy
threshold for each `k`.

Author: OpenAI ChatGPT (2026)
"""

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    logging as hf_logging,
)

# Disable transformers progress bar
hf_logging.set_verbosity_error()


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def load_model(model_name: str = "gpt2-medium"):
    """Load a causal language model and tokenizer, freeze parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def tokenise_text(text: str, tokenizer, max_length: int):
    """Tokenise a string, truncate to `max_length` tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[:max_length]


def build_embeddings(mem_vectors: torch.Tensor, token_ids: torch.Tensor, model):
    """
    Build an embedding tensor of shape (1, k + seq_len, d) from
    memory vectors and token ids.
    """
    token_embeds = model.get_input_embeddings()(token_ids)  # (1, seq_len, d)
    mem_embeds = mem_vectors.unsqueeze(0)  # (1, k, d)
    return torch.cat([mem_embeds, token_embeds], dim=1)


def compute_teacher_forcing_metrics(
    model, tokenizer, token_ids, mem_vectors=None
):
    """
    Teacher‑forcing evaluation.

    Returns (accuracy, correct_tokens, loss_bits_per_token).
    """
    device = next(model.parameters()).device
    seq_len = token_ids.size(1)

    if mem_vectors is None:
        inputs = token_ids
        embeds = None
    else:
        # prepend dummy token ids for the memory positions
        dummy_ids = torch.zeros((1, mem_vectors.size(0)), dtype=torch.long,
                                device=device)
        inputs = torch.cat([dummy_ids, token_ids], dim=1)
        embeds = build_embeddings(mem_vectors, token_ids, model)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=embeds,
            input_ids=inputs if embeds is None else None,
        )
        logits = outputs.logits

    # The model predicts the next token; we ignore the first position of the
    # original sequence (the token immediately following the mem/dummy tokens).
    if mem_vectors is None:
        preds = logits[:, 1:, :]  # (1, seq_len-1, vocab)
        targets = token_ids[:, 1:]  # (1, seq_len-1)
    else:
        preds = logits[:, mem_vectors.size(0) + 1 :, :]  # skip mem + dummy
        targets = token_ids[:, 1:]  # (1, seq_len-1)

    # Accuracy
    predicted_ids = torch.argmax(preds, dim=-1)
    correct = (predicted_ids == targets).sum().item()
    accuracy = correct / targets.numel()
    correct_tokens = correct

    # Cross‑entropy loss (bits per token)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1),
    ).item()
    loss_bits = loss / math.log(2)

    return accuracy, correct_tokens, loss_bits


def train_mem_vectors(
    tokenizer,
    model,
    token_ids,
    mem_k: int,
    lr=5e-3,
    epochs=300,
    device="cpu",
    verbose=False,
):
    """
    Train `mem_k` memory vectors to minimise cross‑entropy of the
    sequence.  Returns a tensor of shape (mem_k, d).
    """
    mem_dim = model.config.n_embd
    mem_vectors = torch.randn((mem_k, mem_dim), requires_grad=True, device=device)
    optimizer = optim.Adam([mem_vectors], lr=lr)

    token_ids = token_ids.to(device)

    for epoch in range(epochs):
        embeds = build_embeddings(mem_vectors, token_ids, model)
        outputs = model(inputs_embeds=embeds)
        logits = outputs.logits[:, 1:, :]  # predict next token
        labels = token_ids[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"[k={mem_k}] Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    return mem_vectors.detach()


def greedy_generate(
    tokenizer,
    model,
    mem_vectors: torch.Tensor,
    max_length: int,
    device="cpu",
    temperature=1.0,
    top_k=5,
):
    """
    Generate text conditioned only on `mem_vectors` (teacher‑forcing style).
    Returns the generated string and the list of token ids.
    """
    device = next(model.parameters()).device
    mem_k = mem_vectors.size(0)
    context_embeds = mem_vectors.unsqueeze(0)  # (1, k, d)

    generated_ids = []

    for _ in range(max_length):
        outputs = model(inputs_embeds=context_embeds)
        logits = outputs.logits[:, -1, :]  # last position
        probs = softmax(logits / temperature, dim=-1)

        topk_vals, topk_ids = torch.topk(probs, top_k)
        topk_probs = topk_vals / topk_vals.sum()
        chosen_id = torch.multinomial(topk_probs, 1).item()
        chosen_id = topk_ids[0, chosen_id].item()

        if chosen_id == tokenizer.eos_token_id:
            break

        generated_ids.append(chosen_id)

        # Append next token embedding to context
        next_emb = model.get_input_embeddings()(
            torch.tensor([[chosen_id]], device=device)
        )
        context_embeds = torch.cat([context_embeds, next_emb], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


def load_random_vocabulary():
    """
    Return a list of ~1000 common English words.
    The list is embedded directly to keep the repo lightweight.
    """
    return [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "big", "little", "red", "blue",
        "green", "yellow", "black", "white", "brown", "gray", "orange",
        "purple", "pink", "silver", "gold", "iron", "steel", "wood",
        "stone", "metal", "glass", "paper", "cloth", "leather", "fabric",
        "rope", "rope", "water", "fire", "earth", "air", "wind", "rain",
        "snow", "storm", "cloud", "sun", "moon", "star", "sky", "sea",
        "river", "mountain", "hill", "valley", "forest", "tree", "bark",
        "leaf", "branch", "root", "flower", "fruit", "seed", "animal",
        "bird", "fish", "cat", "dog", "horse", "cow", "pig", "sheep",
        "goat", "elephant", "tiger", "lion", "bear", "wolf", "fox",
        "rabbit", "deer", "monkey", "snake", "lizard", "frog", "toad",
        "beetle", "ant", "spider", "crab", "shark", "whale", "dolphin",
        "octopus", "squid", "jellyfish", "starfish", "clownfish",
        "salmon", "trout", "bass", "eel", "cod", "tuna", "shrimp",
        "crab", "lobster", "prawn", "clam", "mussel", "scallop",
        "oyster", "snail", "slug", "worm", "maggot", "fly", "mosquito",
        "bee", "wasp", "hornet", "ant", "ladybug", "caterpillar",
        "butterfly", "moth", "dragonfly", "grasshopper", "cricket",
        "beetle", "tick", "mite", "spider", "scorpion", "centipede",
        "millipede", "slug", "worm", "snail", "algae", "plant",
        "flower", "grass", "tree", "bush", "seedling", "sapling",
        "fruit", "vegetable", "fruit", "vegetable", "meat", "fish",
        "drink", "water", "juice", "milk", "bread", "cheese", "butter",
        "oil", "salt", "pepper", "sugar", "cocoa", "coffee", "tea",
        "wine", "beer", "whisky", "rum", "vodka", "gin", "brandy",
        "liqueur", "sake", "cider", "soda", "pop", "cola", "sprite",
        "fanta", "pepsi", "diet", "zero", "diet", "bottle", "can",
        "bowl", "plate", "spoon", "fork", "knife", "napkin", "table",
        "chair", "desk", "couch", "bed", "pillow", "blanket", "sheet",
        "shampoo", "soap", "toothbrush", "toothpaste", "razor",
        "comb", "brush", "hair", "clothes", "shirt", "pants",
        "dress", "skirt", "jacket", "coat", "socks", "shoes",
        "hat", "gloves", "umbrella", "sunglasses", "wallet",
        "phone", "computer", "laptop", "tablet", "camera", "phone",
        "book", "paper", "pen", "pencil", "eraser", "chalk", "marker",
        "box", "bag", "backpack", "suitcase", "ticket", "passport",
        "license", "money", "cash", "coin", "credit", "debit",
        "bank", "store", "mall", "market", "shop", "restaurant",
        "cafe", "bar", "hotel", "hostel", "inn", "airport", "station",
        "bus", "train", "plane", "boat", "ship", "car", "truck",
        "bike", "motorcycle", "bus", "train", "plane", "ship",
        "subway", "metro", "tram", "taxi", "uber", "lyft", "ambulance",
        "fire", "police", "hospital", "clinic", "doctor", "nurse",
        "patient", "medicine", "surgery", "virus", "bacteria",
        "disease", "health", "fitness", "exercise", "sport",
        "game", "music", "movie", "TV", "radio", "news", "journal",
        "blog", "forum", "forum", "internet", "web", "site", "link",
        "page", "home", "profile", "contact", "email", "message",
        "chat", "call", "video", "voice", "sound", "image", "photo",
        "picture", "video", "movie", "film", "animation", "cartoon",
        "comic", "graphic", "drawing", "paint", "art", "design",
        "logo", "brand", "brand", "company", "business", "company",
        "product", "service", "product", "service", "order", "buy",
        "sell", "price", "cost", "value", "money", "budget",
        "finance", "investment", "stock", "bond", "loan", "credit",
        "debt", "interest", "rate", "tax", "taxes", "law", "court",
        "judge", "lawyer", "attorney", "law", "regulation",
        "policy", "policy", "government", "state", "country",
        "city", "town", "village", "neighborhood", "district",
        "region", "area", "land", "country", "nation", "people",
        "population", "demography", "culture", "culture", "tradition",
        "custom", "custom", "history", "history", "past", "future",
        "present", "time", "today", "yesterday", "tomorrow",
        "now", "later", "soon", "later", "moment", "second",
        "minute", "hour", "day", "week", "month", "year",
    ]


def generate_random_sequence(length, vocab, seed=42):
    """
    Generate a random word sequence of `length` words using `vocab`.
    Returns the string and the token ids.
    """
    random.seed(seed)
    words = [random.choice(vocab) for _ in range(length)]
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(args):
    # 1. Load model
    tokenizer, model = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Load or generate texts
    if args.random:
        vocab = load_random_vocabulary()
        texts = [
            generate_random_sequence(length, vocab, seed=seed)
            for seed, length in enumerate(args.lengths)
        ]
    else:
        if args.text_file:
            with open(args.text_file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            texts = [args.text]

    # 3. Process each length and each k
    for length in args.lengths:
        print(f"\n=== Length {length} ===")
        for k in args.k:
            print(f"\n--- Using {k} memory vector(s) ---")
            results = []
            capacity_met = False

            for txt in texts:
                # Truncate to desired length
                token_ids = torch.tensor(
                    [tokenise_text(txt, tokenizer, length)],
                    dtype=torch.long,
                ).unsqueeze(0)

                # Train memory vectors
                mem_vectors = train_mem_vectors(
                    tokenizer,
                    model,
                    token_ids,
                    mem_k=k,
                    lr=args.lr,
                    epochs=args.epochs,
                    device=device,
                    verbose=args.verbose,
                )

                # Teacher‑forcing metrics
                acc_mem, correct_mem, loss_mem = compute_teacher_forcing_metrics(
                    model, tokenizer, token_ids, mem_vectors
                )
                acc_base, correct_base, loss_base = compute_teacher_forcing_metrics(
                    model, tokenizer, token_ids, mem_vectors=None
                )

                token_gain = correct_mem - correct_base
                info_gain = loss_base - loss_mem

                # Free‑generation accuracy
                rec_text, rec_ids = greedy_generate(
                    tokenizer, model, mem_vectors,
                    max_length=length - 1, device=device
                )
                # Compare generated tokens to original
                gen_tokens = torch.tensor(rec_ids, dtype=torch.long)
                orig_tokens = token_ids[0, 1:]  # exclude initial dummy token
                min_len = min(gen_tokens.size(0), orig_tokens.size(0))
                gen_match = (gen_tokens[:min_len] == orig_tokens[:min_len]).sum().item()
                free_acc = gen_match / (length - 1)

                print(f"Reconstructed text: {rec_text}")
                print(f"Teacher‑forcing accuracy: {acc_mem:.4f}")
                print(f"Free‑generation accuracy: {free_acc:.4f}")
                print(f"Token gain: {token_gain}")
                print(f"Information gain (bits/token): {info_gain:.4f}")

                results.append(
                    {
                        "text_length": length,
                        "k": k,
                        "accuracy_tf": acc_mem,
                        "accuracy_free": free_acc,
                        "token_gain": token_gain,
                        "info_gain_bits_per_token": info_gain,
                        "correct_tokens_mem": correct_mem,
                        "correct_tokens_base": correct_base,
                        "loss_bits_base": loss_base,
                        "loss_bits_mem": loss_mem,
                    }
                )

                # Check if threshold met for decoding capacity
                if acc_mem >= args.threshold:
                    capacity_met = True

            # Save results for this length and k
            out_path = Path(args.output_dir) / f"results_len{length}_k{k}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                header = (
                    "text_length,k,accuracy_tf,accuracy_free,token_gain,"
                    "info_gain_bits_per_token,correct_tokens_mem,correct_tokens_base,"
                    "loss_bits_base,loss_bits_mem\n"
                )
                f.write(header)
                for r in results:
                    f.write(
                        f"{r['text_length']},{r['k']},{r['accuracy_tf']:.4f},"
                        f"{r['accuracy_free']:.4f},{r['token_gain']:.0f},"
                        f"{r['info_gain_bits_per_token']:.4f},{r['correct_tokens_mem']},"
                        f"{r['correct_tokens_base']},{r['loss_bits_base']:.4f},"
                        f"{r['loss_bits_mem']:.4f}\n"
                    )
            print(f"Results written to {out_path}")

            # Record decoding capacity for this k
            if capacity_met:
                capacity_k = length
            else:
                capacity_k = 0  # no length reached threshold

            # Write capacity summary
            cap_path = Path(args.output_dir) / "decoding_capacity.csv"
            cap_path.parent.mkdir(parents=True, exist_ok=True)
            if cap_path.exists():
                # Append
                with open(cap_path, "a", encoding="utf-8") as f:
                    f.write(f"{k},{capacity_k}\n")
            else:
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write("k,decoding_capacity\n")
                    f.write(f"{k},{capacity_k}\n")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory Vector Compression Demo – Reproduction of core metrics."
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default="data/snippets.txt",
        help="Path to a text file containing one sentence per line.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog. The dog was not amused.",
        help="Fallback text if --text-file is not provided.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256],
        help="List of token lengths to evaluate.",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="List of numbers of memory vectors to try.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate for memory vector optimisation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of optimisation steps per text.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Accuracy threshold for decoding capacity.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate random word sequences instead of using the text file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store CSV outputs.",
    )
    args = parser.parse_args()

    # Fix seeds
    set_seed(42)
    random.seed(42)
    torch.manual_seed(42)
    main(args)