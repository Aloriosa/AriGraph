"""
Evaluation utilities: compute top‑1 accuracy, LCA distance, and correlations.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, pearsonr
from . import wordnet_utils as wnutils


def evaluate_vm(model: torch.nn.Module,
                loader,
                device: torch.device,
                compute_lca: bool = True):
    """
    Evaluate a vision‑only model (ResNet, etc.) on a dataset loader.
    Returns top‑1 accuracy and mean LCA distance (if requested).
    """
    model.eval()
    correct = 0
    total = 0
    lca_total = 0.0

    with torch.no_grad():
        for imgs, syn_ids in tqdm(loader, desc="Evaluating VM"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total += imgs.size(0)
            pred_indices = preds.cpu().numpy()
            # Convert indices to synset IDs
            pred_syns = [model.class_to_synset[idx] for idx in pred_indices]
            true_syns = syn_ids  # already synset IDs from dataset

            correct += np.sum([int(p == t) for p, t in zip(pred_syns, true_syns)])

            if compute_lca:
                lca_total += np.sum(
                    [wnutils.lca_distance(wnutils.synset_from_id(t),
                                          wnutils.synset_from_id(p))
                     for t, p in zip(true_syns, pred_syns)]
                )

    acc = correct / total
    avg_lca = lca_total / total if compute_lca else None
    return acc, avg_lca


def evaluate_clip(model: torch.nn.Module,
                  preprocess,
                  loader,
                  device: torch.device,
                  compute_lca: bool = True):
    """
    Evaluate a CLIP model on a dataset loader.
    """
    model.eval()
    correct = 0
    total = 0
    lca_total = 0.0

    with torch.no_grad():
        for imgs, syn_ids in tqdm(loader, desc="Evaluating CLIP"):
            # Preprocess images as required by CLIP
            imgs = [preprocess(img) for img in imgs]
            imgs = torch.stack(imgs).to(device)

            logits_per_image, _ = model(imgs, None)  # text is None
            probs = torch.softmax(logits_per_image, dim=1)
            preds = probs.argmax(dim=1)

            total += imgs.size(0)
            pred_indices = preds.cpu().numpy()
            pred_syns = [model.class_to_synset[idx] if hasattr(model, "class_to_synset")
                         else f"n{idx:07d}" for idx in pred_indices]
            true_syns = syn_ids

            correct += np.sum([int(p == t) for p, t in zip(pred_syns, true_syns)])

            if compute_lca:
                lca_total += np.sum(
                    [wnutils.lca_distance(wnutils.synset_from_id(t),
                                          wnutils.synset_from_id(p))
                     for t, p in zip(true_syns, pred_syns)]
                )

    acc = correct / total
    avg_lca = lca_total / total if compute_lca else None
    return acc, avg_lca


def compute_correlations(ids, oods):
    """
    Compute Pearson correlation and R² between ID metrics and OOD accuracy.
    ids and oods are dicts: {model_name: value}
    """
    id_vals = np.array([ids[name] for name in ids])
    ood_vals = np.array([oods[name] for name in ids])
    r2 = r2_score(id_vals, ood_vals)
    pear = pearsonr(id_vals, ood_vals)[0]
    return r2, pear