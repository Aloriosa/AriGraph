from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

# Model paths
MODEL_DIR = "/home/submission/models"

# Load featurizers
class Featurizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading featurizer: {self.model_name}")
        model_path = os.path.join(MODEL_DIR, self.model_name)
        
        # For reproducibility, use local model if available
        if os.path.exists(model_path):
            print(f"Loading local model: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            # Download from HuggingFace
            print(f"Downloading model from HuggingFace: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts to embeddings"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings.cpu().numpy()
        
        return embeddings

class SciBERTFeaturizer(Featurizer):
    def __init__(self):
        super().__init__("allenai/scibert_scivocab_uncased")

class RoBERTaFeaturizer(Featurizer):
    def __init__(self):
        super().__init__("roberta-base")

# Factory
def get_featurizer(dataset_name: str) -> Featurizer:
    """Get appropriate featurizer based on dataset"""
    if dataset_name in ["AQuA-RAT", "SVAMP"]:
        return SciBERTFeaturizer()
    elif dataset_name == "StrategyQA":
        return RoBERTaFeaturizer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}"