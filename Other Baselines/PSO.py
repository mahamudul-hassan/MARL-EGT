"""
run_PSO.py  -- Particle Swarm Optimisation. Run this in its own notebook/process.
Writes PSO_{E3,E4,E7,E8}_*.csv into metaheuristics_results/.
"""
import numpy as np, time
from egt_common import (LOWER, UPPER, N_PARAMS, fitness_for_optimizer,
                        run_search_experiment, N_RUNS)


def pso_run(target_eq, eq_label, run_id,
            n_particles=30, n_iter=40, n_fit_seeds=5,
            w=0.729, c1=1.494, c2=1.494):
    local_rng = np.random.default_rng(run_id * 3571 + 97)
    pos = local_rng.uniform(LOWER, UPPER, size=(n_particles, N_PARAMS))
    vel = local_rng.uniform(-(UPPER-LOWER), (UPPER-LOWER), size=(n_particles, N_PARAMS)) * 0.1
    fits = np.array([fitness_for_optimizer(pos[i], target_eq, n_fit_seeds, base_seed=run_id*10000+i)
                     for i in range(n_particles)])
    pbest_pos = pos.copy(); pbest_fit = fits.copy()
    gbest_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy(); gbest_fit = float(pbest_fit[gbest_idx])

    for it in range(n_iter):
        r1 = local_rng.random((n_particles, N_PARAMS))
        r2 = local_rng.random((n_particles, N_PARAMS))
        vel = w*vel + c1*r1*(pbest_pos - pos) + c2*r2*(gbest_pos - pos)
        v_max = (UPPER - LOWER) * 0.2
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, LOWER, UPPER)
        fits = np.array([fitness_for_optimizer(pos[i], target_eq, n_fit_seeds,
                                                base_seed=run_id*10000+it*1000+i)
                         for i in range(n_particles)])
        improved = fits > pbest_fit
        pbest_pos[improved] = pos[improved].copy(); pbest_fit[improved] = fits[improved]
        gen_best = int(np.argmax(pbest_fit))
        if pbest_fit[gen_best] > gbest_fit:
            gbest_fit = float(pbest_fit[gen_best]); gbest_pos = pbest_pos[gen_best].copy()
        if gbest_fit >= 1.0:
            return gbest_pos, gbest_fit, it + 1
    return gbest_pos, gbest_fit, n_iter


if __name__ == '__main__':
    t0 = time.time()
    run_search_experiment('PSO', pso_run, N_RUNS)
    print(f"\nPSO done in {(time.time()-t0)/60:.1f} min")
