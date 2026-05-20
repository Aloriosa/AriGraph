import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import SimpleUNet
from utils import alpha_beta, dot_alpha_beta

# ------------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 0.1
NUM_STEPS = 50           # Euler steps
DT = 1.0 / NUM_STEPS

# ------------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------------
model = SimpleUNet().to(device)
model.load_state_dict(torch.load('checkpoints/velocity.pth', map_location=device))
model.eval()

# ------------------------------------------------------------------
# 3. Load a single test image
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(
        # Use the first test sample for reproducibility
        torch.randn(1, 28, 28).numpy()
    )
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# For demo, we just generate a random target image
x1 = torch.randn(1, 1, 28, 28, device=device)

# ------------------------------------------------------------------
# 4. Construct base sample x0 = x1 + sigma * noise
# ------------------------------------------------------------------
noise = torch.randn_like(x1) * SIGMA
x0 = x1 + noise

# ------------------------------------------------------------------
# 5. Probability‑flow ODE integration (Euler)
# ------------------------------------------------------------------
x = x0.clone()
for step in range(NUM_STEPS):
    t = torch.tensor([step / NUM_STEPS], device=device)
    b = model(x, t)
    x = x + DT * b

# ------------------------------------------------------------------
# 6. Unnormalize and plot
# ------------------------------------------------------------------
x_cpu = x.squeeze().detach().cpu()
x_cpu = x_cpu * 0.5 + 0.5  # unnormalize to [0,1]
x_cpu = x_cpu.clamp(0, 1)

plt.figure(figsize=(3,3))
plt.imshow(x_cpu.numpy(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('sample.png')
print("Sample image saved to sample.png")