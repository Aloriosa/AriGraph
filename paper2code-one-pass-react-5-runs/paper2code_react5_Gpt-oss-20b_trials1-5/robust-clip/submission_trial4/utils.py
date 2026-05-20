import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm

@torch.no_grad()
def zero_shot_accuracy(model, dataloader, device="cuda"):
    """
    Zero‑shot classification accuracy for CLIP.
    Assumes the dataset has a `classes` attribute listing class names.
    """
    model.eval()
    logits_sum = torch.zeros(len(dataloader.dataset.classes), device=device)
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            logits = model.get_image_features(images)
            # compute similarity with all class text
            with torch.no_grad():
                text_tokens = model.tokenize(dataloader.dataset.classes).to(device)
                text_features = model.get_text_features(text_tokens)
            sims = logits @ text_features.t()  # (B, num_classes)
            probs = F.softmax(sims, dim=-1)
            logits_sum += probs.sum(0)
            total += images.size(0)
    acc = (logits_sum.argmax() == torch.tensor(dataloader.dataset.classes).int()).float().item()
    return acc

def load_clip_model(device="cuda"):
    """
    Load OpenAI CLIP (ViT-B/32) and return the model and its device.
    """
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

def get_vision_encoder(model):
    return model.visual

def get_text_encoder(model):
    return model.encode_text

def get_image_features(model, images):
    return model.encode_image(images)

def get_text_features(model, text_tokens):
    return model.encode_text(text_tokens)