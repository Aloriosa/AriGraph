import numpy as np

def compute_accuracy(predictions, golds):
    """
    Compute accuracy as percentage of correct predictions.
    """
    correct = sum(p == g for p, g in zip(predictions, golds))
    return 100.0 * correct / len(golds)

def evaluate_on_dataset(
    dataset,
    model,
    tokenizer,
    embedding_model,
    device,
    n_samples=10,
    max_new_tokens=256,
):
    """
    Run the full pipeline on a HuggingFace dataset.
    Returns a dict with accuracy for each method.
    """
    preds_self = []
    preds_cpw = []
    preds_scw = []
    golds = []

    for example in tqdm(dataset):
        # Prepare prompt (simple chain‑of‑thought style)
        question = example["question"] if "question" in example else example["text"]
        prompt = f"Question: {question}\nAnswer:"
        # Generate rationales
        rationales = generate_rationales(
            model,
            tokenizer,
            prompt,
            n=n_samples,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        # Extract answers
        answers = [parse_answer(r) for r in rationales]
        # Gold answer (normalize)
        gold = str(example["answer"] if "answer" in example else example["label"])
        golds.append(gold)

        # Baseline self‑consistency
        best, _ = majority_vote(answers)
        preds_self.append(best)

        # CPW
        emb = embedding_model.embed(rationales)
        best_cpw, _ = centroid_proximity_weighting(emb, answers)
        preds_cpw.append(best_cpw)

        # SCW
        best_scw, _ = semantic_consensus_weighting(emb, answers)
        preds_scw.append(best_scw)

    acc_self = compute_accuracy(preds_self, golds)
    acc_cpw = compute_accuracy(preds_cpw, golds)
    acc_scw = compute_accuracy(preds_scw, golds)

    return {
        "self_consistency": acc_self,
        "cpw": acc_cpw,
        "scw": acc_scw,
    }