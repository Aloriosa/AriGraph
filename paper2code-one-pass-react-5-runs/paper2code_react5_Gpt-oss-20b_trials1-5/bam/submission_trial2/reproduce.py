import numpy as np
import os
import csv
from ba_m import BaM
from advi import ADVI
from gsm import GSM
from utils import kl_gaussian, inv

def run_experiments():
    np.random.seed(0)
    results = []

    # Target dimensions
    dims = [4, 16, 64, 256]
    # Batch sizes for BaM
    batch_sizes = [2, 5, 10, 20, 40]
    max_iter = 2000  # number of iterations for all algorithms

    for D in dims:
        # generate random positive definite covariance
        A = np.random.randn(D, D)
        Sigma_target = A @ A.T + 0.1 * np.eye(D)
        mu_target = np.zeros(D)

        # ADVI run
        advi = ADVI(mu_target, Sigma_target, lr_mu=0.01, lr_Sigma=0.01,
                    max_iter=max_iter, seed=0)
        kl_advi, grad_advi = advi.run()
        results.append({
            'alg': 'ADVI',
            'D': D,
            'batch': None,
            'iter': max_iter,
            'kl': kl_advi[-1],
            'grad_evals': grad_advi
        })

        # GSM run
        gsm = GSM(mu_target, Sigma_target, max_iter=max_iter, seed=0)
        kl_gsm, grad_gsm = gsm.run()
        results.append({
            'alg': 'GSM',
            'D': D,
            'batch': None,
            'iter': max_iter,
            'kl': kl_gsm[-1],
            'grad_evals': grad_gsm
        })

        # BaM runs with different batch sizes
        for B in batch_sizes:
            bam = BaM(mu_target, Sigma_target, batch_size=B,
                      lambda_reg=1.0, max_iter=max_iter, seed=0)
            kl_bam, grad_bam = bam.run()
            results.append({
                'alg': 'BaM',
                'D': D,
                'batch': B,
                'iter': max_iter,
                'kl': kl_bam[-1],
                'grad_evals': grad_bam
            })

    # write CSV
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', 'benchmark.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alg', 'D', 'batch', 'iter', 'kl', 'grad_evals'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results written to {csv_path}")

if __name__ == "__main__":
    run_experiments()