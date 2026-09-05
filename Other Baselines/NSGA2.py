"""
run_NSGA2.py  -- NSGA-II. Run this in its own notebook/process.
Writes NSGA-II_{E3,E4,E7,E8}_*.csv into metaheuristics_results/.
"""
import numpy as np, time
from egt_common import (LOWER, UPPER, N_PARAMS, evaluate_robustness,
                        run_search_experiment, N_RUNS)


def nsga2_run(target_eq, eq_label, run_id,
              pop_size=40, n_gen=40, n_fit_seeds=5,
              crossover_prob=0.9, eta_c=15.0, eta_m=20.0):
    local_rng = np.random.default_rng(run_id * 6271 + 41)

    def evaluate_individual(ind, seed_base):
        n_conv, mean_dist, rate = evaluate_robustness(ind, target_eq, n_fit_seeds, base_seed=seed_base)
        return np.array([-rate, mean_dist])

    def sbx_crossover(p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        for d in range(N_PARAMS):
            if local_rng.random() < crossover_prob and abs(p1[d] - p2[d]) > 1e-10:
                y1, y2 = min(p1[d], p2[d]), max(p1[d], p2[d])
                lo, hi = LOWER[d], UPPER[d]
                u = local_rng.random()
                beta = 1.0 + 2*(y1-lo)/(y2-y1)
                alpha = 2.0 - beta**(-(eta_c+1))
                beta_q = (u*alpha)**(1/(eta_c+1)) if u <= 1/alpha else (1/(2-u*alpha))**(1/(eta_c+1))
                c1[d] = 0.5*((p1[d]+p2[d]) - beta_q*(y2-y1))
                beta2 = 1.0 + 2*(hi-y2)/(y2-y1)
                alpha2 = 2.0 - beta2**(-(eta_c+1))
                beta_q2 = (u*alpha2)**(1/(eta_c+1)) if u <= 1/alpha2 else (1/(2-u*alpha2))**(1/(eta_c+1))
                c2[d] = 0.5*((p1[d]+p2[d]) + beta_q2*(y2-y1))
        return np.clip(c1, LOWER, UPPER), np.clip(c2, LOWER, UPPER)

    def polynomial_mutation(ind):
        child = ind.copy()
        for d in range(N_PARAMS):
            if local_rng.random() < 1.0/N_PARAMS:
                lo, hi = LOWER[d], UPPER[d]
                delta_max = hi - lo
                u = local_rng.random()
                delta = (2*u)**(1/(eta_m+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta_m+1))
                child[d] = np.clip(ind[d] + delta*delta_max, lo, hi)
        return child

    def fast_non_dominated_sort(obj_vals):
        n = len(obj_vals)
        dom_count = np.zeros(n, int); dom_set = [[] for _ in range(n)]; fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(obj_vals[i] <= obj_vals[j]) and np.any(obj_vals[i] < obj_vals[j]):
                    dom_set[i].append(j)
                elif np.all(obj_vals[j] <= obj_vals[i]) and np.any(obj_vals[j] < obj_vals[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0: fronts[0].append(i)
        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0: nxt.append(j)
            k += 1; fronts.append(nxt)
        return fronts[:-1]

    def crowding_distance(obj_vals, front):
        n = len(front)
        if n <= 2: return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(obj_vals.shape[1]):
            si = np.argsort(obj_vals[front, m])
            dist[si[0]] = np.inf; dist[si[-1]] = np.inf
            rng_obj = obj_vals[front[si[-1]], m] - obj_vals[front[si[0]], m]
            if rng_obj < 1e-10: continue
            for k in range(1, n-1):
                dist[si[k]] += (obj_vals[front[si[k+1]], m] - obj_vals[front[si[k-1]], m]) / rng_obj
        return dist

    pop = local_rng.uniform(LOWER, UPPER, size=(pop_size, N_PARAMS))
    obj = np.array([evaluate_individual(pop[i], run_id*10000+i) for i in range(pop_size)])
    best_conv_rate = 0.0; best_vec = pop[0].copy()

    for gen in range(n_gen):
        offspring, obj_off = [], []
        while len(offspring) < pop_size:
            idx = local_rng.integers(0, pop_size, 2)
            c1, c2 = sbx_crossover(pop[idx[0]], pop[idx[1]])
            c1, c2 = polynomial_mutation(c1), polynomial_mutation(c2)
            sb = run_id*100000 + gen*1000 + len(offspring)
            offspring.append(c1); obj_off.append(evaluate_individual(c1, sb))
            if len(offspring) < pop_size:
                offspring.append(c2); obj_off.append(evaluate_individual(c2, sb+1))
        combined = np.vstack([pop, np.array(offspring)])
        combined_obj = np.vstack([obj, np.array(obj_off)])
        fronts = fast_non_dominated_sort(combined_obj)
        new_idx = []
        for front in fronts:
            if len(new_idx) + len(front) <= pop_size:
                new_idx.extend(front)
            else:
                needed = pop_size - len(new_idx)
                cd = crowding_distance(combined_obj, front)
                new_idx.extend([front[k] for k in np.argsort(-cd)[:needed]])
                break
        pop = combined[new_idx]; obj = combined_obj[new_idx]
        best_idx = int(np.argmin(obj[:, 0])); rate = -obj[best_idx, 0]
        if rate > best_conv_rate:
            best_conv_rate = rate; best_vec = pop[best_idx].copy()
        if best_conv_rate >= 1.0:
            return best_vec, best_conv_rate, gen + 1
    return best_vec, best_conv_rate, n_gen


if __name__ == '__main__':
    t0 = time.time()
    run_search_experiment('NSGA-II', nsga2_run, N_RUNS)
    print(f"\nNSGA-II done in {(time.time()-t0)/60:.1f} min")
