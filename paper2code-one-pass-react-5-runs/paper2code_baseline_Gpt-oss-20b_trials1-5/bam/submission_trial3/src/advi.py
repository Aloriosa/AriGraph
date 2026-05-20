# src/advi.py
import jax
import jax.numpy as jnp
import optax
from .utils import mvn_logpdf

def advi(mu0, Sigma0, key, B, lr, T, target_logp):
    """Simple ADVI (stochastic ELBO) with Adam."""
    params = {"mu": mu0, "Sigma": Sigma0}
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def loss_fn(p, subkey):
        z = jax.random.multivariate_normal(subkey, p["mu"], p["Sigma"], shape=(B,))
        logp = target_logp(z)
        logq = mvn_logpdf(z, p["mu"], p["Sigma"])
        return -jnp.mean(logp - logq)  # negative ELBO

    @jax.jit
    def step(params, opt_state, subkey):
        grads = jax.grad(loss_fn)(params, subkey)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for t in range(T):
        key, subkey = jax.random.split(key)
        params, opt_state = step(params, opt_state, subkey)

    return params["mu"], params["Sigma"]