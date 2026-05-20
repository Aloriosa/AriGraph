import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler
from torch.nn.utils import spectral_norm

class Adaptor(nn.Module):
    """
    Small residual adaptor that learns a shift for each UNet layer.
    The adaptor is initialized to zero so that the pre‑trained
    UNet behaves unchanged at the start.
    """
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.adaptor_layers = nn.ModuleDict()
        for name, module in unet.named_modules():
            if isinstance(module, nn.Conv2d):
                # we add a small learnable bias (or scaling) per conv layer
                adaptor = nn.Conv2d(module.in_channels, module.out_channels,
                                    kernel_size=module.kernel_size,
                                    stride=module.stride,
                                    padding=module.padding,
                                    bias=True)
                nn.init.constant_(adaptor.bias, 0.0)
                nn.init.zeros_(adaptor.weight)
                self.adaptor_layers[name] = adaptor
        # freeze the original UNet
        for p in unet.parameters():
            p.requires_grad = False

    def forward(self, x, *args, **kwargs):
        # x is the input to the UNet
        # we simply add the adaptor output to the original UNet output
        # by monkey‑patching the forward pass
        return self._forward_with_adaptor(x, *args, **kwargs)

    def _forward_with_adaptor(self, x, *args, **kwargs):
        # custom forward that adds adaptor outputs
        # We rely on the UNet's internal structure:
        # For each conv layer, we add the adaptor output.
        # This is a lightweight implementation that works for the
        # UNet2DModel from diffusers.
        # We clone the original forward and add adaptor bias.
        out = x
        # The UNet2DModel has a forward that iterates over its blocks.
        # We'll override the conv operations by visiting its layers.
        # For simplicity, we call the original forward and then add the
        # adaptor outputs to the intermediate activations.
        # Note: This is a toy implementation; for production, one would
        # subclass UNet2DModel and replace convs.
        out = kwargs['unet'](x, *args, **kwargs)
        # Add adaptor bias to the final output
        for name, adaptor in self.adaptor_layers.items():
            # Find the module in the UNet
            module = kwargs['unet'].get_submodule(name)
            if isinstance(module, nn.Conv2d):
                # compute adaptor output
                adaptor_out = adaptor(out)
                out = out + adaptor_out
        return out

class DiffusionModel(nn.Module):
    """
    Wrapper that contains:
    - a pre‑trained UNet (source model)
    - a DDPM scheduler
    - an adaptor that learns domain shift
    """
    def __init__(self, pretrained_name='google/ddpm-cifar10-32', device='cuda'):
        super().__init__()
        self.device = device
        self.unet = UNet2DModel.from_pretrained(pretrained_name).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_name)
        self.adaptor = Adaptor(self.unet).to(device)

    def train_step(self, images, timesteps, noise, gamma, classifier, step_size, jitter):
        """
        One training step for the adaptor parameters.
        """
        # 1. generate noised images
        noisy_imgs = self.scheduler.add_noise(images, noise, timesteps).to(self.device)
        # 2. obtain predicted noise from UNet + adaptor
        with torch.cuda.amp.autocast():
            pred_noise = self.unet(noisy_imgs, timesteps, encoder_hidden_states=None, return_dict=False)[0]
            # add adaptor adjustment
            pred_noise = pred_noise + self.adaptor(noisy_imgs, timesteps, encoder_hidden_states=None, return_dict=False)[0]
        # 3. similarity term: classifier gradient
        with torch.no_grad():
            logits = classifier(noisy_imgs, timesteps)
            target = torch.ones(logits.shape[0], dtype=torch.long, device=self.device)  # treat target domain as positive
            grad = torch.autograd.grad(logits.sum(), noisy_imgs)[0]  # gradient w.r.t. image
        # 4. loss
        mse = nn.functional.mse_loss(pred_noise, noise)
        sim = -gamma * (pred_noise * grad).mean()
        loss = mse + sim
        # 5. backprop only adaptor
        loss.backward()
        return loss.item()

    def sample(self, num_samples, guidance_scale=1.0):
        """
        Generate images using the fine‑tuned model.
        """
        generator = torch.Generator(device=self.device).manual_seed(0)
        latents = torch.randn((num_samples, self.unet.in_channels, 32, 32), generator=generator, device=self.device)
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            timesteps = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                # UNet prediction
                noise_pred = self.unet(latents, timesteps, encoder_hidden_states=None, return_dict=False)[0]
                noise_pred = noise_pred + self.adaptor(latents, timesteps, encoder_hidden_states=None, return_dict=False)[0]
                # DDPM step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        images = latents.clamp(-1, 1)
        images = (images + 1) / 2  # scale to [0,1]
        return images