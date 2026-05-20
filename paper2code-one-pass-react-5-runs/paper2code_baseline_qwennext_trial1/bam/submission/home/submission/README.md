# Batch and Match (BaM) Reproduction Repository

This repository contains the complete reproduction of the paper:

**"Batch and match: black-box variational inference with a score-based divergence"**

## Overview

This reproduction implements the Batch and Match (BaM) algorithm for black-box variational inference using a score-based divergence as described in the paper.

The key contribution of the paper is a novel approach to variational inference that uses a score-based divergence instead of the traditional evidence lower bound (ELBO). The BaM algorithm alternates between a "batch" step (sampling from the current variational distribution) and a "match" step (updating the variational parameters using closed-form updates that match the score functions.

The implementation includes:
1. The core Batch and Match (BaM) algorithm implementation
2. Comparisons with ADVI (Automatic Differentiation Variational Inference) and GSM (Gaussian Score Matching)
3. Reproduction scripts to generate the results

## Reproduction Instructions

To reproduce the results from the paper, run the following commands: