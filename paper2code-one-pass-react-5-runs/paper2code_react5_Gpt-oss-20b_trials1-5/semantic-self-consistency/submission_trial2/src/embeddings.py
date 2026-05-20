import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

def embed_texts(texts, model_name):
    """
    Compute mean‑pooled BERT‑style embeddings for a list of texts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over tokens
        token_embeds = outputs.last_hidden_state  # (batch, seq_len, dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(token_embeds * attention_mask, dim=1)
        denom = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        batch_embeds = (summed / denom).cpu().numpy()
        embeddings.append(batch_embeds)

    return np.vstack(embeddings)