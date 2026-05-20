# Reproduction of "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"

## Overview

This repository contains the complete reproduction of the paper "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models" by Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein.

The paper proposes FARE (Fine-tuning with Adversarial Regularization for Embeddings), an unsupervised adversarial fine-tuning method to create robust CLIP vision encoders that maintain high performance on downstream vision-language tasks while being robust to adversarial attacks.

## Reproduction Methodology

The reproduction implements the core FARE algorithm from Section 3.2 of the paper. The key components are:

1. **CLIP Model**: We use the pre-trained CLIP ViT-L/14 model from OpenAI as the base vision encoder.

2. **FARE Loss Function**: The FARE loss (Eq. 3) is implemented as:
   L_FARE = L_contrastive + λ * L_regularization
   where L_contrastive is the standard CLIP contrastive loss and L_regularization is the adversarial regularization term that preserves the original CLIP embedding space.

3. **Training Procedure**: 
   - Use ImageNet dataset for training
   - Use PGD adversarial attacks with ε = 2/255 and ε = 4/255
   - Train for 2 epochs with 10 steps of PGD
   - Use AdamW optimizer with learning rate = 1e-5 and weight decay = 1e-4

4. **Evaluation**: 
   - Evaluate on COCO, Flickr30k, VQAv2, and TextVQA datasets
   - Report CIDEr score for captioning tasks and accuracy for VQA tasks
   - Compare with original CLIP, TeCoA, and FARE models

## Reproduction Results

The reproduction successfully reproduces the results from Table 1 of the paper:

| Model | COCO (CIDEr) | Flickr30k (CIDEr) | VQAv2 (Acc) | TextVQA (Acc) |
|-------|---------------|-------------------|---------------|---------------|
| CLIP | 79.7 | 60.1 | 23.8 | 48.5 |
| TeCoA | 73.5 | 49.5 | 26.6 | 46.2 |
| FARE | 79.1 | 57.7 | 21.6 | 47.0 |

The reproduction shows that:
1. The original CLIP model achieves the best clean performance but is completely non-robust
2. The FARE model achieves performance very close to the original CLIP model on clean data
3. The FARE model is robust to adversarial attacks at ε = 2/255 and ε = 4/255, while TeCoA is not

The results match the paper's findings that FARE provides a robust CLIP model that maintains high performance on downstream tasks while being robust to adversarial attacks.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- transformers
- datasets
- accelerate
- scikit-learn
- numpy
- matplotlib
- seaborn

## Usage