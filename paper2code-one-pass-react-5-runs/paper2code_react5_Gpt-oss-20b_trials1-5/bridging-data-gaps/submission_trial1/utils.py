import os
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import lpips
import numpy as np
from torch_fidelity import fid_score

# ---------------------------------------------
#  Data utilities
# ---------------------------------------------
class NoisyImageDataset(Dataset):
    """
    Wraps a base dataset (CIFAR‑10 or SVHN) and returns
    noised images at a random timestep. The label indicates
    whether the image comes from the source (0) or target (1).
    """
    def __init__(self, base_ds, scheduler, label, seed=42):
        self.base_ds = base_ds
        self.scheduler = scheduler
        self.label = label
        random.seed(seed)
        self.t_min = 0
        self.t_max = scheduler.num_train_timesteps - 1

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]  # PIL image
        img = transforms.ToTensor()(img)  # [0,1]
        t = random.randint(self.t_min, self.t_max)
        t_tensor = torch.tensor(t, dtype=torch.long)
        noise = torch.randn_like(img)
        x_t = self.scheduler.add_noise(img, noise, t_tensor)
        return x_t, t_tensor, torch.tensor(self.label, dtype=torch.long)

# ---------------------------------------------
#  Metric utilities
# ---------------------------------------------
def compute_fid(generated_dir: str, real_dir: str, device: torch.device):
    """
    Compute FID using torch‑fidelity.
    """
    return fid_score.calculate_fid_given_paths(
        [generated_dir, real_dir],
        cuda=device.type == "cuda",
        batch_size=32,
        device=device,
        dims=2048,
        num_workers=4
    )[0]

def compute_lpips(generated_dir: str, real_dir: str, device: torch.device):
    """
    Compute average minimum LPIPS distance between each generated image
    and the set of real target images.
    """
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    gen_imgs = sorted([os.path.join(generated_dir, f) for f in os.listdir(generated_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    real_imgs = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Load all real images once
    real_tensors = []
    for f in real_imgs:
        img = torchvision.io.read_image(f).float() / 255.0
        img = torchvision.transforms.functional.resize(img, (32, 32))
        real_tensors.append(img)
    real_tensors = torch.stack(real_tensors).to(device)  # (N, C, H, W)

    lpips_vals = []
    with torch.no_grad():
        for f in tqdm(gen_imgs, desc="LPIPS"):
            g_img = torchvision.io.read_image(f).float() / 255.0
            g_img = torchvision.transforms.functional.resize(g_img, (32, 32))
            g_img = g_img.unsqueeze(0).to(device)  # (1, C, H, W)
            distances = lpips_fn(g_img, real_tensors)  # (1, N)
            min_dist = distances.min().item()
            lpips_vals.append(min_dist)
    return np.mean(lpips_vals)