"""
bo_replicator.py
==============================================================================
Bayesian Optimization (Optuna / TPE) outer search whose fitness is the
REPLICATOR-SIMULATION convergence rate -- the SAME objective GA / PSO / NSGA-II
use in this project -- instead of an inner PPO rollout.

Each proposed 13-parameter vector is scored by
    egt_common.fitness_for_optimizer(vec, target_eq, n_seeds)
which runs n_seeds replicator simulations and returns the fraction that converge
to the target equilibrium. BO maximizes that rate. No PPO, no JAX.

Import this from the per-equilibrium runner files (run_BO_E3.py, ... E8.py).
Each file runs ONE equilibrium for n_runs and writes the same CSV schema
(FIELDNAMES) that the metaheuristic runs produce, so BO output is directly
comparable to GA / PSO / NSGA-II output in metaheuristics_results/.

Requires: egt_common.py in the same folder, and `optuna` installed.
"""

import os
import time
import csv
import numpy as np
import pandas as pd
import optuna

import egt_common as egt
from egt_common import (
    LOWER, UPPER, N_PARAMS, PARAM_NAMES, EQUILIBRIA,
    N_RUNS, N_FIT_SEEDS, N_VAL_SEEDS, OUTPUT_DIR, FIELDNAMES,
    fitness_for_optimizer, validate_optimal_params, vec_to_dict,
    _summarise, save_eigenvalue_distributions,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# BO single-run search. Signature matches the algo_fn contract used by the
# metaheuristics:  algo_fn(target_eq, eq_label, run_id, n_fit_seeds)
#                     -> (best_vec, best_fit, n_evals_used)
# One Optuna study per run; each trial = one replicator-fitness evaluation.
# ---------------------------------------------------------------------------
def bo_run(target_eq, eq_label, run_id,
           n_fit_seeds=N_FIT_SEEDS, n_bo_trials=200, master_seed=42):
    """One BO study over the 13 game parameters, maximizing the replicator
    convergence rate. Mirrors ga_run / pso_run bookkeeping."""
    lo = np.asarray(LOWER, float)
    hi = np.asarray(UPPER, float)

    state = {'best_fit': -np.inf, 'best_vec': None, 'i': 0}

    def objective(trial):
        vec = np.array(
            [trial.suggest_float(PARAM_NAMES[d], lo[d], hi[d]) for d in range(N_PARAMS)],
            dtype=float,
        )
        # replicator convergence rate in [0,1] -- identical fitness to GA/PSO/NSGA-II
        fit = fitness_for_optimizer(
            vec, target_eq, n_seeds=n_fit_seeds,
            base_seed=run_id * 100000 + state['i'],
        )
        if fit > state['best_fit']:
            state['best_fit'] = float(fit)
            state['best_vec'] = vec.copy()
        state['i'] += 1
        return fit  # maximize

    sampler = optuna.samplers.TPESampler(seed=int(master_seed) + run_id)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_bo_trials, show_progress_bar=False)

    # If nothing was recorded (shouldn't happen), fall back to study best.
    if state['best_vec'] is None:
        bp = study.best_params
        state['best_vec'] = np.array([bp[n] for n in PARAM_NAMES], dtype=float)
        state['best_fit'] = float(study.best_value)

    return state['best_vec'], state['best_fit'], state['i']


# ---------------------------------------------------------------------------
# Single-equilibrium driver. Same body as egt_common.run_search_experiment but
# for ONE equilibrium, so each runner file handles exactly one of E3/E4/E7/E8.
# Reuses egt_common's validation, eigenvalue test, CSV schema and summary.
# ---------------------------------------------------------------------------
def run_bo_single_eq(eq_key, n_runs=N_RUNS, n_bo_trials=200, master_seed=42,
                     algo_name='BO'):
    if eq_key not in EQUILIBRIA:
        raise ValueError(f"Unknown eq_key {eq_key!r}; expected one of {list(EQUILIBRIA)}")
    eq_cfg = EQUILIBRIA[eq_key]
    target_eq = eq_cfg['target']
    eq_label = eq_cfg['label']

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, f'{algo_name}_{eq_key}_all_runs.csv')
    summary_csv = os.path.join(OUTPUT_DIR, f'{algo_name}_{eq_key}_summary.csv')

    print(f"\n{'='*70}\n  {algo_name} (replicator fitness)  |  {eq_key} {eq_cfg['name']}  "
          f"target={target_eq}  ({eq_cfg['type']})\n{'='*70}")
    print(f"  runs={n_runs}  BO trials/run={n_bo_trials}  fit seeds={N_FIT_SEEDS}  "
          f"val seeds={N_VAL_SEEDS}")

    rows = []
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for run_id in range(1, n_runs + 1):
            t0 = time.time()
            best_vec, best_fit, n_used = bo_run(
                target_eq, eq_label, run_id=run_id,
                n_fit_seeds=N_FIT_SEEDS, n_bo_trials=n_bo_trials,
                master_seed=master_seed,
            )
            val = validate_optimal_params(best_vec, target_eq, eq_label, run_id, N_VAL_SEEDS)
            wall = time.time() - t0
            p_dict = vec_to_dict(best_vec)
            row = {
                'run_id': run_id, 'algorithm': algo_name, 'equilibrium': eq_key,
                'search_best_fitness': round(best_fit, 6), 'search_n_gen_used': n_used,
                **{n: round(float(p_dict[n]), 6) for n in PARAM_NAMES},
                'val_conv_rate': round(val['conv_rate_validation'], 6),
                'val_n_conv': val['n_conv_validation'],
                'val_mean_dist': round(val['mean_final_dist'], 6),
                'val_mean_conv_time': round(val['mean_conv_time'], 4),
                'stable': val['stable'],
                'all_eigenvalues_negative': val['all_eigenvalues_negative'],
                'all_eigenvalues_strong': val['all_eigenvalues_strong'],
                'lambda_x': round(val['lambda_x'], 6),
                'lambda_y': round(val['lambda_y'], 6),
                'lambda_z': round(val['lambda_z'], 6),
                'wall_time_s': round(wall, 2),
            }
            rows.append(row); writer.writerow(row); f.flush()
            status = '+' if val['conv_rate_validation'] == 1.0 else ('~' if val['conv_rate_validation'] > 0 else 'x')
            print(f"  Run {run_id:3d}/{n_runs} {status}  fit={best_fit:.2f}  "
                  f"conv={val['conv_rate_validation']:.2f}  "
                  f"allNegEig={val['all_eigenvalues_negative']}  t={wall:.1f}s")

    df = pd.DataFrame(rows)
    summary = _summarise(df, algo_name, eq_key)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    save_eigenvalue_distributions(algo_name, eq_key, df)
    print(f"  -> {summary['pct_fully_converged']}% fully converged | "
          f"{summary['pct_all_eig_negative']}% all-eig-negative | "
          f"mean wall {summary['mean_wall_time_s']}s")
    return summary
