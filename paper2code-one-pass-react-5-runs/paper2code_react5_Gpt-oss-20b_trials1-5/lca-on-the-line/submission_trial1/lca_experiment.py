#!/usr/bin/env python3
"""
LCA‑on‑the‑Line Simplified Demo on CIFAR‑10
------------------------------------------
Evaluates several vision and vision‑language models on CIFAR‑10, computes the
Lowest Common Ancestor (LCA) distance using the official WordNet hierarchy,
constructs a latent hierarchy via K‑means clustering, creates a noisy OOD split,
and reports the correlation between ID LCA distance and OOD accuracy.

Author: ChatGPT (OpenAI)
License: MIT
"""

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
# 1. WordNet hierarchy construction
# --------------------------------------------------------------------------- #
import nltk
from nltk.corpus import wordnet as wn

# CIFAR‑10 class names (used for mapping to WordNet synsets)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Manual mapping from CIFAR‑10 classes to WordNet synset IDs
# (synset names are of the form 'word.n.01')
CLASS_TO_SYNS = {
    "airplane": "airplane.n.01",
    "automobile": "car.n.01",
    "bird": "bird.n.01",
    "cat": "cat.n.01",
    "deer": "deer.n.01",
    "dog": "dog.n.01",
    "frog": "frog.n.01",
    "horse": "horse.n.01",
    "ship": "ship.n.01",
    "truck": "truck.n.01",
}


class WordNetHierarchy:
    """
    Builds a tree representation of the WordNet hierarchy for the given
    synset set.  Each node has an integer ID, a parent ID (None for root),
    and a probability (sum of leaf probabilities below it).
    """

    def __init__(self, synset_ids):
        # synset_ids: list of synset IDs (strings) for the leaves
        self.synset_ids = synset_ids
        self.node_id = {}  # synset string -> integer ID
        self.parent = {}  # node ID -> parent ID
        self.children = defaultdict(list)  # parent ID -> list of child IDs
        self.prob = {}  # node ID -> probability
        self.info = {}  # node ID -> -log2(prob)
        self._build_tree()

    def _build_tree(self):
        # Assign IDs to leaves first
        current_id = 0
        for syn_id in self.synset_ids:
            self.node_id[syn_id] = current_id
            current_id += 1

        # Build parent relations by walking hypernyms until root
        for syn_id in self.synset_ids:
            node = syn_id
            parent = self._first_hypernym(node)
            while parent is not None:
                if parent not in self.node_id:
                    self.node_id[parent] = current_id
                    current_id += 1
                # Record parent-child
                child_id = self.node_id[node]
                parent_id = self.node_id[parent]
                self.parent[child_id] = parent_id
                self.children[parent_id].append(child_id)
                # Move up
                node = parent
                parent = self._first_hypernym(node)

        # Root nodes have no parent
        all_nodes = set(self.node_id.values())
        children_nodes = set(self.parent.keys())
        root_nodes = all_nodes - children_nodes
        for root_id in root_nodes:
            self.parent[root_id] = None

        # Compute probabilities (leaf probability = 1 / num_leaves)
        num_leaves = len(self.synset_ids)
        leaf_prob = 1.0 / num_leaves

        # Initialize leaf probabilities
        for syn_id in self.synset_ids:
            leaf_id = self.node_id[syn_id]
            self.prob[leaf_id] = leaf_prob

        # Propagate probabilities upward
        visited = set()

        def accumulate(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            # Sum over children if any
            if node_id in self.children:
                total = 0.0
                for child_id in self.children[node_id]:
                    accumulate(child_id)
                    total += self.prob[child_id]
                self.prob[node_id] = total

        # Start from leaves
        for leaf_id in [self.node_id[s] for s in self.synset_ids]:
            accumulate(leaf_id)

        # Compute information content
        for node_id, p in self.prob.items():
            self.info[node_id] = -math.log2(p) if p > 0 else 0.0

    def _first_hypernym(self, syn_id):
        """Return the first hypernym synset string of syn_id, or None if root."""
        syn = wn.synset(syn_id)
        hyper = syn.hypernyms()
        if not hyper:
            return None
        return hyper[0].name()

    def lca(self, node_a, node_b):
        """Return the lowest common ancestor node ID for node_a and node_b."""
        ancestors_a = set()
        cur = node_a
        while cur is not None:
            ancestors_a.add(cur)
            cur = self.parent[cur]
        cur = node_b
        while cur not in ancestors_a:
            cur = self.parent[cur]
        return cur

    def lca_distance(self, node_a, node_b):
        """
        LCA distance as defined in the paper:
        D = I(y) - I(LCA(y, y'))
        where I(.) is the information content of a node.
        """
        anc = self.lca(node_a, node_b)
        return self.info[node_a] - self.info[anc]


# --------------------------------------------------------------------------- #
# 2. Data loaders
# --------------------------------------------------------------------------- #
def get_cifar10_dataloaders(batch_size=256):
    """Return DataLoader for CIFAR‑10 train and test sets."""
    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    transform = T.Compose([T.ToTensor(), normalize])

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader, test_set


def noisy_dataloader(original_loader, noise_std=0.5):
    """
    Return a DataLoader that adds Gaussian noise to each image.
    """
    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, orig_ds):
            self.ds = orig_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, tgt = self.ds[idx]
            noise = torch.randn_like(img) * noise_std
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
            return noisy_img, tgt

    noisy_ds = NoisyDataset(original_loader.dataset)
    return DataLoader(
        noisy_ds,
        batch_size=original_loader.batch_size,
        shuffle=False,
        num_workers=2,
    )


# --------------------------------------------------------------------------- #
# 3. Model utilities
# --------------------------------------------------------------------------- #
def load_resnet18(num_classes=10, pretrained=True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    # Replace final fully‑connected layer
    in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_ftrs, num_classes)
    return model


def load_resnet50(num_classes=10, pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_ftrs, num_classes)
    return model


def load_mobilenetv2(num_classes=10, pretrained=True):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    in_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_ftrs, num_classes)
    return model


def load_clip_model(device):
    """
    Load OpenAI CLIP ViT‑B/32 zero‑shot model.
    """
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess


# --------------------------------------------------------------------------- #
# 4. Feature extraction for latent hierarchy
# --------------------------------------------------------------------------- #
def extract_class_avg_features(train_loader, device):
    """
    Use a pretrained ResNet‑18 (ImageNet) to extract 512‑dim features
    from the CIFAR‑10 training set and compute the average per class.
    """
    feature_extractor = torchvision.models.resnet18(pretrained=True)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    # Remove the final fully‑connected layer
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.to(device)

    class_sums = [torch.zeros(512, device=device) for _ in range(10)]
    class_counts = [0] * 10

    with torch.no_grad():
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs).squeeze(-1).squeeze(-1)  # shape [B, 512]
            for feat, lab in zip(feats, labs):
                class_sums[lab] += feat
                class_counts[lab] += 1

    class_means = [s / c for s, c in zip(class_sums, class_counts)]
    class_means_np = np.stack([m.cpu().numpy() for m in class_means])
    return class_means_np


# --------------------------------------------------------------------------- #
# 5. Latent hierarchy construction
# --------------------------------------------------------------------------- #
def build_latent_hierarchy(class_vectors, max_level):
    """
    Build a hierarchical clustering of the class vectors.
    Returns:
        cluster_assignments: dict level -> array of cluster ids for each class
        cluster_sizes: dict level -> dict cluster_id -> size
    """
    n_classes = class_vectors.shape[0]
    cluster_assignments = {}
    cluster_sizes = {}
    for level in range(1, max_level + 1):
        k = min(2**level, n_classes)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(class_vectors)
        cluster_assignments[level] = labels
        size_dict = {}
        for c in labels:
            size_dict[c] = size_dict.get(c, 0) + 1
        cluster_sizes[level] = size_dict
    # Root cluster (level 0)
    cluster_assignments[0] = np.zeros(n_classes, dtype=int)
    cluster_sizes[0] = {0: n_classes}
    return cluster_assignments, cluster_sizes


def compute_latent_lca_matrix(cluster_assignments, cluster_sizes):
    """
    Compute the LCA distance matrix for the latent hierarchy.
    """
    n = len(cluster_assignments[1])
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Find deepest level where assignments match
            deepest = max(
                l
                for l in cluster_assignments
                if cluster_assignments[l][i] == cluster_assignments[l][j]
            )
            cluster_id = cluster_assignments[deepest][i]
            size = cluster_sizes[deepest][cluster_id]
            dist[i, j] = math.log2(size)
    return dist


# --------------------------------------------------------------------------- #
# 6. WordNet LCA matrix construction
# --------------------------------------------------------------------------- #
def compute_wordnet_lca_matrix(hierarchy, class_to_node):
    """
    Compute the LCA distance matrix for the WordNet hierarchy.
    """
    n = len(class_to_node)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            node_i = class_to_node[i]
            node_j = class_to_node[j]
            dist[i, j] = hierarchy.lca_distance(node_i, node_j)
    return dist


# --------------------------------------------------------------------------- #
# 7. Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, labs in loader:
        imgs = imgs.to(device)
        labs = labs.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    print(f"  Epoch {epoch:02d} – loss: {avg_loss:.4f}")


def fine_tune(model, train_loader, val_loader, device, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
    # Evaluate after training
    acc, _, _ = evaluate(model, val_loader, device)
    print(f"  Fine‑tuned validation accuracy: {acc:.4f}")


# --------------------------------------------------------------------------- #
# 8. Evaluation utilities
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    """Return accuracy, predictions, and labels."""
    model.eval()
    correct = 0
    total = 0
    preds = []
    labels = []

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            _, pred = probs.max(1)
            correct += (pred == labs).sum().item()
            total += labs.size(0)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(labs.cpu().numpy().tolist())

    accuracy = correct / total
    return accuracy, preds, labels


def compute_mean_lca(preds, labels, dist_matrix):
    distances = [dist_matrix[p][l] for p, l in zip(preds, labels)]
    return np.mean(distances), np.std(distances)


# --------------------------------------------------------------------------- #
# 9. CLIP inference helper
# --------------------------------------------------------------------------- #
def evaluate_clip(
    clip_model,
    preprocess,
    dataset,
    device,
    noisy=False,
    noise_std=0.5,
):
    """
    Zero‑shot CLIP inference on the given dataset.
    Returns accuracy, predictions, and labels.
    """
    clip_model.eval()
    class_texts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    with torch.no_grad():
        # Pre‑compute text embeddings
        text_tokens = clip.tokenize(class_texts).to(device)
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    preds = []
    labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    for imgs, labs in loader:
        if noisy:
            noise = torch.randn_like(imgs) * noise_std
            imgs = imgs + noise
            imgs = torch.clamp(imgs, 0.0, 1.0)

        imgs = imgs.to(device)
        labs = labs.to(device)
        image_embeddings = clip_model.encode_image(imgs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        logits = (image_embeddings @ text_embeddings.T).float()
        probs = logits.softmax(dim=-1)
        _, pred = probs.max(1)

        correct += (pred == labs).sum().item()
        total += labs.size(0)
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(labs.cpu().numpy().tolist())

    accuracy = correct / total
    return accuracy, preds, labels


# --------------------------------------------------------------------------- #
# 10. Main script
# --------------------------------------------------------------------------- #
def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build WordNet hierarchy
    synset_ids = [CLASS_TO_SYNS[c] for c in CIFAR10_CLASSES]
    hierarchy = WordNetHierarchy(synset_ids)

    # Map class indices to node IDs
    class_to_node_wordnet = {
        idx: hierarchy.node_id[CLASS_TO_SYNS[c]] for idx, c in enumerate(CIFAR10_CLASSES)
    }

    # Get data loaders
    train_loader, test_loader, test_set = get_cifar10_dataloaders(batch_size=args.batch_size)

    # Create noisy OOD loader
    noisy_loader = noisy_dataloader(test_loader, noise_std=args.noise_std)

    # --------------------------------------------------------------
    # Build latent hierarchy
    # --------------------------------------------------------------
    print("\nExtracting class‑average features for latent hierarchy...")
    class_vecs = extract_class_avg_features(train_loader, device)
    max_level = int(math.ceil(math.log2(len(CIFAR10_CLASSES))))
    cluster_assignments, cluster_sizes = build_latent_hierarchy(class_vecs, max_level)
    cluster_assignments, cluster_sizes  # silence unused variable warning
    latent_lca_matrix = compute_latent_lca_matrix(cluster_assignments, cluster_sizes)

    # WordNet LCA matrix
    wordnet_lca_matrix = compute_wordnet_lca_matrix(hierarchy, class_to_node_wordnet)

    # --------------------------------------------------------------
    # Prepare models
    # --------------------------------------------------------------
    models = {}
    # Vision models
    models["ResNet‑18"] = load_resnet18(num_classes=10, pretrained=True)
    models["ResNet‑50"] = load_resnet50(num_classes=10, pretrained=True)
    models["MobileNet‑V2"] = load_mobilenetv2(num_classes=10, pretrained=True)

    # Fine‑tune vision models
    print("\nFine‑tuning vision models on CIFAR‑10...")
    for name, mdl in models.items():
        print(f"\n=== {name} ===")
        mdl.to(device)
        fine_tune(mdl, train_loader, test_loader, device, epochs=args.epochs)

    # Vision‑Language model (CLIP)
    print("\nLoading CLIP ViT‑B/32 zero‑shot model...")
    clip_model, clip_preprocess = load_clip_model(device)
    clip_model.eval()
    models["CLIP ViT‑B/32"] = clip_model  # we'll handle inference separately

    # --------------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------------
    results = []

    for name, mdl in models.items():
        print(f"\n=== Evaluating {name} ===")
        if name.startswith("CLIP"):
            # CLIP zero‑shot inference
            acc_id, preds_id, labels_id = evaluate_clip(
                mdl, clip_preprocess, test_set, device, noisy=False
            )
            acc_ood, preds_ood, labels_ood = evaluate_clip(
                mdl, clip_preprocess, test_set, device, noisy=True, noise_std=args.noise_std
            )
        else:
            # Vision models
            acc_id, preds_id, labels_id = evaluate(mdl, test_loader, device)
            acc_ood, preds_ood, labels_ood = evaluate(mdl, noisy_loader, device)

        # LCA statistics (WordNet)
        id_lca_wordnet, id_lca_std_wordnet = compute_mean_lca(
            preds_id, labels_id, wordnet_lca_matrix
        )
        ood_lca_wordnet, ood_lca_std_wordnet = compute_mean_lca(
            preds_ood, labels_ood, wordnet_lca_matrix
        )

        # LCA statistics (latent)
        id_lca_latent, id_lca_std_latent = compute_mean_lca(
            preds_id, labels_id, latent_lca_matrix
        )
        ood_lca_latent, ood_lca_std_latent = compute_mean_lca(
            preds_ood, labels_ood, latent_lca_matrix
        )

        print(f"  ID Accuracy: {acc_id:.4f}")
        print(f"  ID Mean LCA (WordNet): {id_lca_wordnet:.4f} ± {id_lca_std_wordnet:.4f}")
        print(f"  ID Mean LCA (Latent): {id_lca_latent:.4f} ± {id_lca_std_latent:.4f}")
        print(f"  OOD Accuracy: {acc_ood:.4f}")
        print(f"  OOD Mean LCA (WordNet): {ood_lca_wordnet:.4f} ± {ood_lca_std_wordnet:.4f}")
        print(f"  OOD Mean LCA (Latent): {ood_lca_latent:.4f} ± {ood_lca_std_latent:.4f}")

        results.append(
            {
                "model": name,
                "id_acc": acc_id,
                "id_lca_wordnet": id_lca_wordnet,
                "id_lca_latent": id_lca_latent,
                "ood_acc": acc_ood,
                "ood_lca_wordnet": ood_lca_wordnet,
                "ood_lca_latent": ood_lca_latent,
            }
        )

    # Correlation analysis across models
    id_lca_wordnet_vals = np.array([r["id_lca_wordnet"] for r in results])
    id_lca_latent_vals = np.array([r["id_lca_latent"] for r in results])
    ood_acc_vals = np.array([r["ood_acc"] for r in results])

    pearson_wordnet = np.corrcoef(id_lca_wordnet_vals, ood_acc_vals)[0, 1]
    pearson_latent = np.corrcoef(id_lca_latent_vals, ood_acc_vals)[0, 1]

    print("\n=== Correlation across models ===")
    print(f"Pearson r (ID LCA WordNet vs OOD accuracy) = {pearson_wordnet:.4f}")
    print(f"Pearson r (ID LCA Latent vs OOD accuracy) = {pearson_latent:.4f}")

    # Save results
    with open("id_results_wordnet.txt", "w") as f:
        f.write("model,accuracy,mean_lca\n")
        for r in results:
            f.write(f"{r['model']},{r['id_acc']:.4f},{r['id_lca_wordnet']:.4f}\n")
    with open("ood_results_wordnet.txt", "w") as f:
        f.write("model,accuracy,mean_lca\n")
        for r in results:
            f.write(f"{r['model']},{r['ood_acc']:.4f},{r['ood_lca_wordnet']:.4f}\n")
    with open("id_results_latent.txt", "w") as f:
        f.write("model,accuracy,mean_lca\n")
        for r in results:
            f.write(f"{r['model']},{r['id_acc']:.4f},{r['id_lca_latent']:.4f}\n")
    with open("ood_results_latent.txt", "w") as f:
        f.write("model,accuracy,mean_lca\n")
        for r in results:
            f.write(f"{r['model']},{r['ood_acc']:.4f},{r['ood_lca_latent']:.4f}\n")
    with open("correlation_wordnet.txt", "w") as f:
        f.write(f"Pearson r: {pearson_wordnet:.4f}\n")
    with open("correlation_latent.txt", "w") as f:
        f.write(f"Pearson r: {pearson_latent:.4f}\n")


# --------------------------------------------------------------------------- #
# 11. Argument parsing
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified LCA‑on‑the‑Line demo on CIFAR‑10."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation (default: 256)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="Standard deviation of Gaussian noise added for OOD (default: 0.5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for fine‑tuning vision models (default: 5)",
    )
    args = parser.parse_args()
    main(args)