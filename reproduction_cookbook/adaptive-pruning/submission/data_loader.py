import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import json
import os

class APTDataset(Dataset):
    """
    Dataset class for APT training.
    Supports various NLP tasks including classification and generation.
    """
    
    def __init__(self, data_path: str, tokenizer, task: str, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.data = []
        
        # Load data based on task
        if task in ['sst2', 'mrpc', 'cola', 'rte', 'qnli', 'mnli', 'qqp']:
            self._load_glue_data(data_path)
        elif task in ['cnndm', 'xsum']:
            self._load_summarization_data(data_path)
        elif task == 'alpaca':
            self._load_alpaca_data(data_path)
        else:
            raise ValueError(f"Unsupported task: {task}")
            
    def _load_glue_data(self, data_path: str):
        """Load GLUE dataset data"""
        # For simplicity, we'll use a mock implementation
        # In practice, this would load from the actual GLUE dataset files
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('\t')
                if self.task == 'sst2':
                    # SST-2: sentence, label
                    if len(parts) >= 2:
                        text = parts[0]
                        label = int(parts[1])
                        self.data.append({'text': text, 'label': label})
                elif self.task == 'mnli':
                    # MNLI: sentence1, sentence2, label
                    if len(parts) >= 3:
                        text1 = parts[0]
                        text2 = parts[1]
                        label = int(parts[2])
                        self.data.append({'text1': text1, 'text2': text2, 'label': label})
                # Add other GLUE tasks as needed
                
    def _load_summarization_data(self, data_path: str):
        """Load summarization dataset data"""
        # For simplicity, we'll use a mock implementation
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    summary = parts[1]
                    self.data.append({'text': text, 'summary': summary})
                    
    def _load_alpaca_data(self, data_path: str):
        """Load Alpaca dataset data"""
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output = item.get('output', '')
                
                if input_text:
                    text = f"{instruction}\n{input_text}"
                else:
                    text = instruction
                    
                self.data.append({'text': text, 'output': output})
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task in ['sst2', 'mrpc', 'cola', 'rte', 'qnli', 'mnli', 'qqp']:
            if self.task == 'mnli':
                # MNLI has two sentences
                text1 = item['text1']
                text2 = item['text2']
                label = item['label']
                
                # Tokenize
                encoded = self.tokenizer(
                    text1, text2,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            else:
                # Single sentence tasks
                text = item['text']
                label = item['label']
                
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
                
        elif self.task in ['cnndm', 'xsum']:
            # Summarization tasks
            text = item['text']
            summary = item['summary']
            
            # Tokenize input
            encoded_input = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Tokenize target
            encoded_target = self.tokenizer(
                summary,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded_input['input_ids'].squeeze(0),
                'attention_mask': encoded_input['attention_mask'].squeeze(0),
                'labels': encoded_target['input_ids'].squeeze(0)
            }
            
        elif self.task == 'alpaca':
            # Instruction-following tasks
            text = item['text']
            output = item['output']
            
            # Create prompt
            prompt = f"### Instruction:\n{text}\n\n### Response:\n{output}"
            
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': encoded['input_ids'].squeeze(0)
            }
            
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        """Get DataLoader for this dataset"""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        
class DataCollator:
    """
    Custom data collator for APT training.
    """
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples):
        # Handle different task types
        if 'text1' in examples[0]:  # MNLI
            texts1 = [example['text1'] for example in examples]
            texts2 = [example['text2'] for example in examples]
            labels = [example['label'] for example in examples]
            
            encoded = self.tokenizer(
                texts1, texts2,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long)
            }
            
        elif 'text' in examples[0] and 'summary' in examples[0]:  # Summarization
            texts = [example['text'] for example in examples]
            summaries = [example['summary'] for example in examples]
            
            encoded_input = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            encoded_target = self.tokenizer(
                summaries,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded_input['input_ids'],
                'attention_mask': encoded_input['attention_mask'],
                'labels': encoded_target['input_ids']
            }
            
        elif 'text' in examples[0] and 'output' in examples[0]:  # Alpaca
            texts = [example['text'] for example in examples]
            outputs = [example['output'] for example in examples]
            
            prompts = []
            for text, output in zip(texts, outputs):
                prompt = f"### Instruction:\n{text}\n\n### Response:\n{output}"
                prompts.append(prompt)
                
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': encoded['input_ids']
            }
            
        else:  # Single sentence classification
            texts = [example['text'] for example in examples]
            labels = [example['label'] for example in examples]
            
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long)
            }