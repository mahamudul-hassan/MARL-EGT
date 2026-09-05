"""
run_CMAES.py  -- CMA-ES. Run this in its own notebook/process.
Writes CMA-ES_{E3,E4,E7,E8}_*.csv into metaheuristics_results/.
"""
import numpy as np, time
from egt_common import (LOWER, UPPER, N_PARAMS, fitness_for_optimizer,
                        run_search_experiment, N_RUNS)


def cmaes_run(target_eq, eq_label, run_id,
              pop_size=30, n_gen=40, n_fit_seeds=5, sigma0=0.3):
    local_rng = np.random.default_rng(run_id * 5237 + 61)
    D = N_PARAMS
    lam = pop_size if pop_size else 4 + int(3 * np.log(D))
    mu  = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1)); weights /= weights.sum()
    mu_eff = 1.0 / (weights**2).sum()
    cc  = (4 + mu_eff/D) / (D + 4 + 2*mu_eff/D)
    cs  = (mu_eff + 2) / (D + mu_eff + 5)
    c1  = 2.0 / ((D + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((D+2)**2 + mu_eff))
    damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(D+1)) - 1) + cs
    chiN = D**0.5 * (1 - 1/(4*D) + 1/(21*D**2))

    def norm(v): return (v - LOWER) / (UPPER - LOWER)
    def denorm(v): return v * (UPPER - LOWER) + LOWER

    mean  = norm(local_rng.uniform(LOWER, UPPER))
    sigma = sigma0; C = np.eye(D); pc = np.zeros(D); ps = np.zeros(D)
    B, D_eig = np.eye(D), np.ones(D); eigeneval = 0
    best_fit = 0.0; best_vec = denorm(np.clip(mean, 0, 1))

    for gen in range(n_gen):
        if gen - eigeneval > lam / (c1+cmu) / D / 10:
            C = np.triu(C) + np.triu(C, 1).T
            D_eig2, B = np.linalg.eigh(C)
            D_eig = np.sqrt(np.maximum(D_eig2, 1e-20)); eigeneval = gen
        zs = local_rng.standard_normal((lam, D))
        ys = (B * D_eig) @ zs.T
        xs_norm = (mean[:, None] + sigma * ys).T
        xs_phys = np.clip(denorm(np.clip(xs_norm, 0, 1)), LOWER, UPPER)
        fits = np.array([fitness_for_optimizer(xs_phys[k], target_eq, n_fit_seeds,
                                                base_seed=run_id*100000+gen*1000+k)
                         for k in range(lam)])
        order = np.argsort(-fits); best_k = int(order[0])
        if fits[best_k] > best_fit:
            best_fit = float(fits[best_k]); best_vec = xs_phys[best_k].copy()
        old_mean = mean.copy()
        mean = weights @ np.clip(xs_norm[order[:mu]], 0, 1)
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (B @ ((mean-old_mean)/sigma / D_eig))
        hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(D+1))
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mu_eff) * ((mean-old_mean)/sigma)
        artmp = (xs_norm[order[:mu]] - old_mean) / sigma
        C = ((1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
             + cmu*(artmp.T @ np.diag(weights) @ artmp))
        sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        sigma = np.clip(sigma, 1e-10, 1.0)
        if best_fit >= 1.0:
            return best_vec, best_fit, gen + 1
    return best_vec, best_fit, n_gen


if __name__ == '__main__':
    t0 = time.time()
    run_search_experiment('CMA-ES', cmaes_run, N_RUNS)
    print(f"\nCMA-ES done in {(time.time()-t0)/60:.1f} min")
