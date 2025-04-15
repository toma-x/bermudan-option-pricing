import time
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import jaxlib
from tqdm import trange

r = 0.05
sigma = 0.2
K = 100
S0 = 100
m = 3
T = 1
t = jnp.linspace(0, T, m+1)
n_v = 2 # normal and antithetic

@jax.jit
def X(key, x, dt):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey)
    increment_pos = jnp.exp((r - sigma**2 / 2) * dt + sigma * jnp.sqrt(dt) * z)
    increment_neg = jnp.exp((r - sigma**2 / 2) * dt - sigma * jnp.sqrt(dt) * z)
    new_states = jnp.array([x * increment_pos, x * increment_neg])
    return key, new_states

@jax.jit
def p_eu(_T, x):
    d1 = (jnp.log(x / K) + (r + (sigma ** 2) / 2) * _T) / (sigma * jnp.sqrt(_T))
    d2 = d1 - sigma * jnp.sqrt(_T)
    return K * jnp.exp(-r * _T) * norm.cdf(-d2) - x * norm.cdf(-d1)

@jax.jit
def h(m, x):
    return jnp.maximum(K - x, 0)

def random_tree_low(key, x, i, t, b, m, n_v, evals=0):
    european_price = p_eu(t[-1] - t[i], x)
    h_value = h(m, x)
    
    if i == m - 1:
        return jnp.maximum(h_value, european_price), 1
    
    def single_step(k):
        k, new_states = X(k, x, t[i + 1] - t[i])
        subkeys = jax.random.split(k, n_v)
        values, new_evals = jax.vmap(lambda sk, new_x: random_tree_low(sk, new_x, i + 1, t, b, m, n_v, evals+1))(subkeys, new_states)
        return jnp.nanmean(values), jnp.sum(new_evals)
    
    def no_exercise(_):
        value, batch_evals = single_step(key)
        return value, 1 + jnp.sum(batch_evals)
    
    def proceed(_):
        keys = jax.random.split(key, b // n_v)
        values, batch_new_evals = jax.vmap(single_step)(keys)
        mask = (jnp.sum(values) - values)/(b-1) <= h_value
        continuation_value = jnp.nanmean(h_value * mask + values * ~mask)
        return jnp.maximum(h_value, continuation_value), 1 + jnp.sum(batch_new_evals)
    
    return jax.lax.cond(
        (i > 0) & (h_value < european_price),
        no_exercise,
        proceed,
        operand=None
    )

def monte_carlo_tree_low(S0, b, n, m, n_v, n_chunks=1):
    start = time.time()
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, n)
    batch_size = max(1, n // n_chunks)

    results_list = []
    evals_list = []

    try:
        for i in trange(0, n, batch_size, desc=f"{b=} {n=}"):
            batch_keys = keys[i: i + batch_size]
            batch_results, batch_evals = jax.vmap(lambda k: random_tree_low(k, S0, 0, t, b, m, n_v))(batch_keys)
            results_list.append(batch_results)
            evals_list.append(batch_evals)

        results = jnp.concatenate(results_list)
        evals = jnp.concatenate(evals_list)

    except jaxlib.xla_extension.XlaRuntimeError as error:
        if batch_size == 1:
            return
        total_bytes_needed = int(str(error).split()[-2])
        memory_info = jax.devices()[0].memory_stats()
        available_bytes = 0.9 * memory_info['bytes_limit'] - memory_info['bytes_in_use']
        n_chunks = 4 * (int(total_bytes_needed // available_bytes) + 1)

        print(f"Retrying with {n_chunks} chunks")
        return monte_carlo_tree_low(S0, b, n, m, n_v, n_chunks)

    n_evals = jnp.mean(evals)
    pruned = 1 - n_evals / ((b**m - 1) / (b - 1))

    mean = jnp.mean(results)
    std = jnp.std(results, ddof=1)
    z = norm.ppf(0.975)
    CI = (mean - z * std / jnp.sqrt(n), mean + z * std / jnp.sqrt(n))
    elapsed = time.time() - start

    return mean, CI, jnp.sum(evals), pruned, elapsed

if b := input():
    b = int(b)
    assert not b % n_v, f"b must be a multiple of {n_v}"
    key = jax.random.PRNGKey(1)
    value, evals = random_tree_low(key, S0, 0, t, b, m, n_v)
    print(f"{b} {value:.3f} {evals:,} evals")
else:
    n = 10_000
    b_values = [10, 20, 50, 100, 200, 500, 1_000, 2_000]
    print("|$b$|$n$|Low Estimate|95% CI|95% CI length|Nodes evaluated|Pruned (%)|Time (s)|")
    print("|-|-|-|-|-|-|-|-|")
    for b in b_values:
        b = b // n_v * n_v
        mean, CI, evals, pruned, elapsed = monte_carlo_tree_low(S0, b, n, m, n_v)
        print(f"|{b:,}|{n:,}|{mean:.3f}|{tuple(map(lambda x: round(x.item(), 3), CI))}|{CI[1]-CI[0]:.3f}|{evals:,}|{100*pruned:.2f}|{elapsed:.1f}|")
