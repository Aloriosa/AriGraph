import json
from torch.utils.data import Dataset

class SimpleQADataset(Dataset):
    """
    Simple QA dataset loaded from a JSONL file.
    Each line must contain 'question' and 'answer' fields.
    """
    def __init__(self, path):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]