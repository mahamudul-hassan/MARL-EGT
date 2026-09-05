"""
run_GA.py  -- Genetic Algorithm. Run this in its own notebook/process.
Writes GA_{E3,E4,E7,E8}_*.csv into metaheuristics_results/.
"""
import numpy as np, time
from egt_common import (LOWER, UPPER, N_PARAMS, fitness_for_optimizer,
                        run_search_experiment, N_RUNS)


def ga_run(target_eq, eq_label, run_id,
           pop_size=30, n_gen=40, n_fit_seeds=5,
           crossover_prob=0.8, mutation_prob=0.15, mutation_scale=0.1):
    local_rng = np.random.default_rng(run_id * 7919 + 13)
    pop = local_rng.uniform(LOWER, UPPER, size=(pop_size, N_PARAMS))
    fits = np.array([fitness_for_optimizer(ind, target_eq, n_fit_seeds, base_seed=run_id*1000+j)
                     for j, ind in enumerate(pop)])
    best_vec = pop[np.argmax(fits)].copy()
    best_fit = float(np.max(fits))

    for gen in range(n_gen):
        new_pop = [best_vec.copy()]            # elitism
        while len(new_pop) < pop_size:
            def tournament():
                idx = local_rng.integers(0, pop_size, 3)
                return pop[idx[np.argmax(fits[idx])]].copy()
            p1, p2 = tournament(), tournament()
            if local_rng.random() < crossover_prob:
                mask = local_rng.random(N_PARAMS) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            for d in range(N_PARAMS):
                if local_rng.random() < mutation_prob:
                    rng_range = (UPPER[d] - LOWER[d]) * mutation_scale
                    child[d] += local_rng.normal(0, rng_range)
            new_pop.append(np.clip(child, LOWER, UPPER))
        pop = np.array(new_pop)
        fits = np.array([fitness_for_optimizer(ind, target_eq, n_fit_seeds,
                                                base_seed=run_id*1000+gen*100+j)
                         for j, ind in enumerate(pop)])
        gen_best_idx = int(np.argmax(fits))
        if fits[gen_best_idx] > best_fit:
            best_fit = float(fits[gen_best_idx]); best_vec = pop[gen_best_idx].copy()
        if best_fit >= 1.0:
            return best_vec, best_fit, gen + 1
    return best_vec, best_fit, n_gen


if __name__ == '__main__':
    t0 = time.time()
    run_search_experiment('GA', ga_run, N_RUNS)
    print(f"\nGA done in {(time.time()-t0)/60:.1f} min")
