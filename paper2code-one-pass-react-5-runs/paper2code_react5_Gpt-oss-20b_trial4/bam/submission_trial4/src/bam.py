# ---------------------------------------------------------------------------

def advi(
    key: jax.random.PRNGKey,
    target_log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    target_mean: jnp.ndarray,
    target_cov: jnp.ndarray,
    T: int,
    B: int,
    lr: float,
    mu_init: jnp.ndarray,
    Sigma_init: jnp.ndarray,
    kl_history: bool = True,
    lr_schedule: Sequence[float] | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simple ADVI implementation using the reparameterisation trick.

    Parameters
    ----------
    key : PRNGKey
        Random key for reproducibility.
    target_log_prob : function
        log p(z) – (unnormalised) log‑density of the target.
    target_score : function
        ∇_z log p(z).
    target_mean, target_cov : ground truth Gaussian parameters (for diagnostics).
    T : int
        Number of iterations.
    B : int
        Batch size.
    lr : float
        Base learning rate for the Adam optimiser.
    mu_init, Sigma_init : initial variational parameters.
    kl_history : bool
        Store KL(p||q_t) per iteration.
    lr_schedule : optional sequence of float
        If provided, overrides the constant learning rate; must have length T.
    """
    D = mu_init.shape[0]
    key, subk = random.split(key)

    # Parameters: mu (D,) and lower‑triangular L (D,D) of Sigma
    def pack_params(mu, L):
        return jnp.concatenate([mu, L.reshape(-1)])

    def unpack_params(packed):
        mu = packed[:D]
        L = packed[D:].reshape(D, D)
        return mu, L

    # Initialise
    L0 = jnp.linalg.cholesky(Sigma_init)
    params = pack_params(mu_init, L0)

    # Optimiser
    if lr_schedule is None:
        lr_schedule = [lr] * T
    else:
        assert len(lr_schedule) == T
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = opt.init(params)

    kl_hist: List[float] = []

    @jax.jit
    def elbo_grad(params, key):
        mu, L = unpack_params(params)
        Sigma = L @ L.T
        z = _sample_gaussian(key, mu, Sigma, B)  # (B, D)
        log_p = vmap(target_log_prob)(z)  # (B,)
        log_q = vmap(_log_q_gaussian, in_axes=(0, None, None))(z, mu, Sigma)
        elbo = jnp.mean(log_p - log_q)  # ELBO
        return -elbo  # we minimise negative ELBO

    for t in range(T):
        key, subk = random.split(subk)
        loss, grads = jax.value_and_grad(elbo_grad)(params, subk)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        mu, L = unpack_params(params)
        Sigma = L @ L.T

        if kl_history:
            kl_hist.append(_kl_gaussian(target_mean, target_cov,
                                        mu, Sigma))

    return mu, Sigma, jnp.array(kl_hist) if kl_history else jnp.array([])