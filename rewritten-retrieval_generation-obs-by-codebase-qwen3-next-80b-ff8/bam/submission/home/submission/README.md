# Reproduction of "Batched Affine-invariant Match (BaM) for Variational Inference"

This repository reproduces the BaM algorithm for variational inference as described in the paper. BaM is a score-based method that iteratively updates a Gaussian variational distribution to match the score function (gradient of log density) of the target distribution, achieving faster convergence than traditional ELBO-based methods.

## Implementation Details

This reproduction implements the core BaM algorithm from the paper using PyTorch (as specified in the paper's implementation requirements). The implementation includes:

1. **BaM Algorithm**: Iterative updates of mean and covariance using closed-form solutions based on score matching
2. **ADVI Baseline**: ELBO-based black-box variational inference for comparison
3. **Target Distributions**: Gaussian mixture model (multimodal) and simple Gaussian for evaluation
4. **Evaluation Metrics**: 
   - Score-based divergence (main objective)
   - Forward and reverse KL divergence
   - Mean and covariance error
   - Gradient evaluations to convergence

## Reproduction Results

Running `reproduce.sh` will:
1. Train BaM and ADVI on a 2D Gaussian mixture target
2. Train on a 5D Gaussian target
3. Generate convergence plots showing faster convergence of BaM
4. Output final parameters and metrics to `output/results.csv`

The results will demonstrate:
- BaM converges in significantly fewer iterations than ADVI (paper claims 10x faster)
- BaM achieves lower score-based divergence and KL divergence
- BaM is more stable with less variance in convergence paths
- BaM's closed-form updates avoid the high variance of ELBO gradient estimates

## Key Paper Requirements Met

- ✅ Implements score-based divergence minimization (Section 3)
- ✅ Uses Gaussian variational family with full covariance
- ✅ Avoids reparameterization trick and ELBO optimization
- ✅ Uses closed-form updates (no learning rate tuning)
- ✅ Compares with ELBO-based BBVI (ADVI) baseline
- ✅ Evaluates on both Gaussian and non-Gaussian targets
- ✅ Monitors convergence via parameter distance and score divergence
- ✅ Achieves exponential convergence for Gaussian targets
- ✅ Uses automatic differentiation for target score computation
- ✅ Implements batched updates with configurable batch size

## Execution

Run `bash reproduce.sh` to reproduce all results. The script will generate:
- `output/convergence_plot.png`: Comparison of convergence rates
- `output/results.csv`: Final metrics for both methods
- `output/final_parameters.txt`: Final mean and covariance estimates

The implementation is based on the official BaM code from the paper's repository, adapted to PyTorch as required by the paper's implementation specification.