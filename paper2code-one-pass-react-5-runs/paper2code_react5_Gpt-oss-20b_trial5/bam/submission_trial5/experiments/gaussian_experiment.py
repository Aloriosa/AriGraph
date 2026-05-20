#!/usr/bin/env python3
# ------------------------------------------------------------------
# Gaussian target experiment comparing BaM, ADVI, GSM
# ------------------------------------------------------------------
import numpy as np
import time
from src.bam import BaM
from src.advi import ADVI
from src.gsm import GSM
from src.utils import kl_gaussian

def gaussian_score(z, mu_star, Sigma_star):
    """Score of Gaussian target: Σ^{-1} (μ - z)."""
    inv = np.linalg.inv(Sigma_star)
    return inv @ (mu_star - z)

def run_dim(D, B_vals, T=200):
    # Target: N(0, I)
    mu_star = np.zeros(D)
    Sigma_star = np.eye(D)

    print(f"\nDimension: {D}")
    for B in B_vals:
        # Initialisation
        mu0 = np.random.uniform(-0.1, 0.1, size=D)
        Sigma0 = np.eye(D)

        # BaM
        bam = BaM(D, mu0, Sigma0, B=B, lambda_reg=B*D, T=T)
        bam_kls = []
        def bam_cb(t, mu, Sigma):
            kl = kl_gaussian(mu, Sigma, mu_star, Sigma_star)
            bam_kls.append(kl)
        bam.run(lambda z: gaussian_score(z, mu_star, Sigma_star), callback=bam_cb)

        # ADVI
        advi = ADVI(D, mu0, Sigma0, B=B, lr=0.01, T=T)
        advi_kls = []
        def advi_cb(mu, Sigma):
            kl = kl_gaussian(mu, Sigma, mu_star, Sigma_star)
            advi_kls.append(kl)
        advi.run(callback=advi_cb)

        # GSM (B=1)
        gsm = GSM(D, mu0, Sigma0, T=T)
        gsm_kls = []
        def gsm_cb(mu, Sigma):
            kl = kl_gaussian(mu, Sigma, mu_star, Sigma_star)
            gsm_kls.append(kl)
        gsm.run(lambda z: gaussian_score(z, mu_star, Sigma_star), callback=gsm_cb)

        print(f"  Batch size B={B}")
        print(f"    BaM final KL : {bam_kls[-1]:.4f}")
        print(f"    ADVI final KL: {advi_kls[-1]:.4f}")
        print(f"    GSM final KL : {gsm_kls[-1]:.4f}")

if __name__ == "__main__":
    dims = [4, 16, 64, 256]
    B_vals = [20, 50, 200]
    for D in dims:
        run_dim(D, B_vals, T=200)