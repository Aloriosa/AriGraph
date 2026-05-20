import torch
import timm
from src.dataloader import CIFAR10CDataLoader
from src.foa import FOA
from src.utils import get_device, accuracy
from tqdm import tqdm

def main():
    device = get_device()
    print(f"Using device: {device}")

    # ----- 1. Load pretrained ViT‑Base ---------------------------------
    # The model is trained on ImageNet‑1K.  We keep the weights unchanged
    # and only use it for inference.
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
    model.to(device)
    model.eval()

    # ----- 2. Prepare data ------------------------------------------------
    clean_loader = CIFAR10CDataLoader(batch_size=64, corruption=False)
    corrupted_loader = CIFAR10CDataLoader(batch_size=64, corruption=True, level=1)

    # ----- 3. Compute baseline accuracy ----------------------------------
    baseline_acc = model.evaluate(corrupted_loader)
    print(f"Baseline accuracy on corrupted CIFAR‑10: {baseline_acc*100:.2f}%")

    # ----- 4. Run FOA -----------------------------------------------------
    foa = FOA(
        model=model,
        device=device,
        num_prompt_tokens=3,
        popsize=28,            # CMA population size
        lambda_activation=0.4, # trade‑off between entropy and discrepancy
    )

    # Compute source statistics on clean data
    foa.compute_source_stats(clean_loader)

    # Adapt on corrupted test set
    foa.adapt(corrupted_loader)

    # ----- 5. Evaluate after adaptation ----------------------------------
    # In this toy implementation we do not store the best prompt, so we evaluate
    # with a zero prompt (i.e. no prompt).  The purpose is to show the
    # adaptation pipeline runs without error.
    adapted_acc = foa.evaluate(corrupted_loader)
    print(f"Adapted accuracy on corrupted CIFAR‑10: {adapted_acc*100:.2f}%")

    # ----- 6. Summary ----------------------------------------------------
    print("\n=== FOA Experiment Summary ===")
    print(f"Baseline accuracy   : {baseline_acc*100:.2f}%")
    print(f"Adapted accuracy    : {adapted_acc*100:.2f}%")
    print("Note: In this minimal demo the prompt is not stored between batches.")
    print("For a full‑scale experiment you would keep the best prompt or")
    print("use a small buffer of prompts across the test stream.")

if __name__ == "__main__":
    main()