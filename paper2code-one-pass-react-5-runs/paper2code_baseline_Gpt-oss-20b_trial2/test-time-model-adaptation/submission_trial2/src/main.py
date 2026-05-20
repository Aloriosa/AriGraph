import argparse
import torch
import timm
from tqdm import tqdm
from src.dataset import ImageNetC
from src.foa import FOA, PromptedViT
from src.utils import accuracy, mean_std, moving_average


def compute_source_stats(model: torch.nn.Module,
                         loader: torch.utils.data.DataLoader,
                         device: torch.device,
                         num_samples: int = 32) -> tuple:
    """
    Compute mean and std of CLS tokens for each block on source data.
    """
    model.eval()
    cls_tokens_per_layer = [[] for _ in range(len(model.blocks))]

    with torch.no_grad():
        count = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # Forward with hooks
            def hook_factory(idx):
                def hook(module, input, output):
                    cls_tokens_per_layer[idx].append(output[:, 0].cpu())
                return hook

            hooks = []
            for idx, blk in enumerate(model.blocks):
                h = blk.register_forward_hook(hook_factory(idx))
                hooks.append(h)

            _ = model(imgs)  # forward

            for h in hooks:
                h.remove()

            count += imgs.size(0)
            if count >= num_samples:
                break

    means = []
    stds = []
    for tokens in cls_tokens_per_layer:
        tokens = torch.cat(tokens, dim=0)  # (N, D)
        m, s = mean_std(tokens, dim=0)
        means.append(m)
        stds.append(s)
    return means, stds


def main():
    parser = argparse.ArgumentParser(description="FOA Reproduction")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ImageNet‑C validation folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-prompt", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.4)
    parser.add_argument("--popsize", type=int, default=28)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load base ViT‑Base
    print("Loading base ViT‑Base...")
    base_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    base_model.eval()
    base_model.to(device)

    # 2. Load source (validation) dataset for statistics
    print("Computing source statistics...")
    src_loader = torch.utils.data.DataLoader(
        ImageNetC(args.dataset, level=5),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    source_means, source_stds = compute_source_stats(
        base_model, src_loader, device, num_samples=32
    )

    # 3. Prepare test loader (ImageNet‑C)
    print("Preparing test loader...")
    test_loader = torch.utils.data.DataLoader(
        ImageNetC(args.dataset, level=5),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 4. Initialize FOA
    foa = FOA(
        model=base_model,
        source_stats=(source_means, source_stds),
        device=device,
        num_prompt=args.num_prompt,
        lambda_=args.lambda_,
        popsize=args.popsize,
        num_generations=args.num_generations,
        gamma=1.0,
    )

    # 5. Run adaptation and evaluate
    print("\n=== FOA on ImageNet‑C (severity 5) ===")
    acc = foa.evaluate(test_loader)
    print(f"\nOverall Accuracy on ImageNet‑C: {acc * 100:.1f}%")

if __name__ == "__main__":
    main()