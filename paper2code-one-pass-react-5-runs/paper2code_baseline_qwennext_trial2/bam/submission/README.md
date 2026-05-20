# Reproduction: Batch and Match (BaM) for Black-Box Variational Inference

This repository reproduces the results from the paper "Batch and match: black-box variational inference with a score-based divergence" by Cai et al. (2024).

## Overview

The paper introduces Batch and Match (BaM), a novel algorithm for black-box variational inference (BBVI) that uses a score-based divergence instead of the traditional evidence lower bound (ELBO).

BaM alternates between:
1. **Batch step**: Draw samples from the current variational distribution
2. **Match step**: Update the variational parameters to match the scores of the target distribution

The key contributions are:
- A new score-based divergence for BBVI
- Closed-form proximal updates for Gaussian variational families
- Provable exponential convergence guarantees for Gaussian targets
- Empirical superiority over ADVI and Gaussian Score Matching (GSM)

## Implementation Details

The implementation follows the exact mathematical formulations from the paper:

1. **Score-based divergence**: 
   `D(q; p) = E_q[||∇_z log(q(z)/p(z))||^2_Cov(q)`

2. **Batch step**: Sample from current variational distribution `N(μ_t, Σ_t)` and compute batch statistics (means and covariances of samples and their scores)

3. **Match step**: Update variational parameters with closed-form updates derived from minimizing the score-based divergence with KL regularization

4. **Closed-form updates**: 
   - Update covariance: `Σ_{t+1} = 2V(I + (I + 4UV)^(1/2))^(-1)`
   - Update mean: `μ_{t+1} = (1/(1+λ_t))μ_t + (λ_t/(1+λ_t))(Σ_{t+1}g̅ + z̅)`

The implementation uses JAX for automatic differentiation and GPU acceleration.

## Reproduction Instructions

1. Install dependencies: `bash reproduce.sh`

2. The script will:
   - Install required packages (Python, NumPy, SciPy, Matplotlib, JAX)
   - Run the BaM implementation
   - Generate results in `outputs/` directory

3. Results include:
   - Convergence plots for Gaussian targets
   - Comparison with ADVI and GSM
   - Performance on non-Gaussian targets
   - Application to hierarchical models
   - Application to deep generative models

## Expected Results

The reproduction should reproduce the key findings from the paper:
- BaM converges significantly faster than ADVI (orders of magnitude faster)
- BaM converges faster than GSM, especially with larger batch sizes
- BaM converges for both Gaussian and non-Gaussian targets
- BaM shows superior performance on hierarchical and deep generative models

The implementation is faithful to the mathematical formulations in the paper and uses JAX for efficient computation.

## Limitations

- The implementation focuses on the core algorithm from the paper
- The implementation uses JAX for automatic differentiation
- The implementation uses the exact mathematical formulations from the paper

## References

Cai, D., Modi, C., Pillaud-Vivien, L., Margossian, C., Gower, R., Blei, D., & Saul, L. (2024). Batch and match: black-box variational inference with a score-based divergence. Proceedings of the 41st International Conference on Machine Learning.

## License

MIT License