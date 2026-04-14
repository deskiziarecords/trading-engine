import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(4,))
def schur_optimize(
    q_total: float,           # Q_t from position sizing
    venues: jax.Array,        # m venues, shape (m, 3) = [liquidity, latency, fees]
    ofi_matrix: jax.Array,    # (m, m) cross-venue OFI correlations
    prev_weights: jax.Array,  # w_{t-1} for recursive slippage
    params: dict              # Static parameters
) -> tuple[jax.Array, dict]:
    """
    Adelic-Schur routing optimization.
    
    Returns: (weights w_t, diagnostics)
    """
    m = venues.shape[0]
    eps = 1e-6
    
    # Unpack parameters
    gamma = params['slippage_gamma']      # Liquidity sensitivity
    delta = params['slippage_delta']      # Convexity (1.5 typical)
    lambda_dist = params['correlation_decay']
    rho_limit = params['adelic_rho_limit']
    alpha_schur = params['alpha_schur']   # 1.5 for adelic tube
    blowup_kappa = params['blowup_kappa'] # 3.0 * ATR multiplier
    
    # 1. Construct cost matrix C
    def slippage(i, w_i):
        q_i = q_total * w_i
        return gamma[i] * jnp.power(jnp.abs(q_i), delta[i])
    
    # Vectorized slippage diagonal
    s_diag = jax.vmap(slippage)(jnp.arange(m), prev_weights)
    
    # Correlated impact: distance-based decay
    # venues[:, 1] = latency proxy for "distance"
    dist_matrix = jnp.abs(venues[:, None, 1] - venues[None, :, 1])
    rho_matrix = jnp.exp(-lambda_dist * dist_matrix)
    
    # OFI-modulated correlation
    C = s_diag[:, None] + s_diag[None, :] + rho_matrix * ofi_matrix * q_total
    
    # Symmetrize and stabilize
    C_tilde = 0.5 * (C + C.T) + eps * jnp.eye(m)
    
    # 2. Schur decomposition (via eig for stability in JAX)
    # Note: JAX doesn't have native Schur, use eig for real symmetric
    eigenvalues, eigenvectors = jnp.linalg.eigh(C_tilde)
    
    # 3. Find minimum eigenvalue (stablest direction)
    k_star = jnp.argmin(eigenvalues)
    lambda_min = eigenvalues[k_star]
    v_opt = eigenvectors[:, k_star]
    
    # 4. Adelic tube validation on eigenvalues
    # Treat eigenvalues as "pricing" on adelic manifold
    def adelic_check_eigen(val):
        # Simplified: check |val^alpha| < rho across simulated p-adic metrics
        log_val = jnp.log(jnp.abs(val) + eps)
        return log_val < jnp.log(rho_limit)  # Archimedean proxy
    
    adelic_valid = jax.vmap(adelic_check_eigen)(eigenvalues).all()
    
    # 5. Blow-up detection
    atr_threshold = venues[:, 2].mean() * blowup_kappa  # Fee as ATR proxy
    no_blowup = lambda_min < atr_threshold
    
    # 6. Simplex projection
    def project_simplex(v):
        """Duchi et al. efficient projection"""
        u = jnp.sort(v)[::-1]  # Descending
        cssv = jnp.cumsum(u) - 1.0  # Cumulative sum minus 1
        ind = jnp.arange(1, m + 1)
        cond = u - cssv / ind > 0
        rho = jnp.sum(cond)  # Number of positive elements
        theta = cssv[rho - 1] / rho
        return jnp.maximum(v - theta, 0.0)
    
    w_raw = project_simplex(v_opt)
    
    # 7. Apply constraints
    valid = adelic_valid & no_blowup & (lambda_min < 0)  # Negative = cost minimum
    
    w_final = w_raw * valid + prev_weights * (1 - valid)  # Revert to prev if invalid
    
    # Renormalize
    w_final = w_final / (w_final.sum() + eps)
    
    diagnostics = {
        'eigenvalues': eigenvalues,
        'lambda_min': lambda_min,
        'k_star': k_star,
        'adelic_valid': adelic_valid,
        'no_blowup': no_blowup,
        'cost_estimate': w_final @ C_tilde @ w_final,
        'concentration': jnp.max(w_final),  # Max venue weight (risk check)
        'entropy': -jnp.sum(w_final * jnp.log(w_final + eps))  # Diversification
    }
    
    return w_final, diagnostics


# Integration with position sizing
def full_execution_pipeline(
    equity: float,
    ev_t: float,
    atr_t: float,
    phi_t: float,
    venues: jax.Array,
    ofi_matrix: jax.Array,
    prev_weights: jax.Array,
    params: dict
):
    """Complete: Size → Route → Execute"""
    
    # 1. Position sizing (from previous)
    q_total, size_diag = position_size(
        equity, ev_t, atr_t, params['atr_ref'], phi_t,
        adelic_valid=True, params=params['sizing']
    )
    
    # 2. Schur routing
    weights, route_diag = schur_optimize(
        q_total, venues, ofi_matrix, prev_weights, params['routing']
    )
    
    # 3. Final quantities per venue
    quantities = q_total * weights
    
    return {
        'total_notional': q_total,
        'venue_weights': weights,
        'venue_quantities': quantities,
        'sizing_diagnostics': size_diag,
        'routing_diagnostics': route_diag,
        'expected_slippage': route_diag['cost_estimate'],
        'execution_entropy': route_diag['entropy']
    }
