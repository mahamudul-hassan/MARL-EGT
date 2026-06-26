"""
egt_common.py
=============================================================================
Shared engine for the metaheuristics-vs-RL comparison. Every model script
(GA, NSGA-II, PSO, CMA-ES, RL) imports from here so the replicator engine,
validation, and eigenvalue test are byte-identical across notebooks.

Each model script runs INDEPENDENTLY and writes its own CSVs into
OUTPUT_DIR. Run them in parallel notebooks; then run merge_results.py.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import warnings, os, time, csv
warnings.filterwarnings('ignore')

# -- Reproducibility ----------------------------------------------------------
MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)

# -- Paths --------------------------------------------------------------------
RL_CSV_DIR      = '.'
RL_CSV_TEMPLATE = '{eq}_optimal_parameters.csv'   # e.g. E3_optimal_parameters.csv
OUTPUT_DIR      = 'metaheuristics_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1.  GAME MODEL
# =============================================================================
PARAM_NAMES = ['P','C_m','B','alpha','F','beta','W_m','I','R_c','C_c','gamma','R_p','C_p']

PARAM_BOUNDS = {
    'P':     (100, 300),
    'C_m':   (50,  200),
    'B':     (5,   20),
    'alpha': (0.5, 1.0),
    'F':     (50,  200),
    'beta':  (0.1, 0.8),
    'W_m':   (20,  100),
    'I':     (5,   50),
    'R_c':   (100, 300),
    'C_c':   (5,   30),
    'gamma': (0.3, 0.9),
    'R_p':   (5,   30),
    'C_p':   (10,  100),
}

LOWER = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES], dtype=float)
UPPER = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES], dtype=float)
N_PARAMS = len(PARAM_NAMES)


def vec_to_dict(v):
    return {n: float(v[i]) for i, n in enumerate(PARAM_NAMES)}


def calc_fitness_differentials(p, x, y, z):
    x, y, z = np.clip([x, y, z], 1e-4, 1 - 1e-4)

    PF1 =  p['P'] - p['C_m']
    PF2 =  p['P'] - p['C_m']
    PF3 = -p['B']
    PF4 = -p['B']
    PF5 =  p['alpha']*p['P'] - p['C_m'] - p['F']
    PF6 =  p['alpha']*p['P'] - p['beta']*p['C_m'] - p['W_m']
    PF7 = -p['F'] - p['B']
    PF8 = -p['B'] - p['F'] - p['I']

    PC1 =  p['R_c']
    PC2 =  p['R_c']
    PC3 = -p['C_c']
    PC4 = -p['C_c']
    PC5 =  p['R_c']
    PC6 =  p['gamma'] * p['R_c']
    PC7 = -p['C_c']
    PC8 =  p['I'] - p['C_c']

    PG1 =  p['R_p'] - p['C_p']
    PG2 =  p['R_p']
    PG3 = -p['C_p']
    PG4 = -p['C_p']
    PG5 =  p['F'] + p['R_p'] - 2*p['C_p']
    PG6 =  p['R_p']
    PG7 =  p['F'] - 2*p['C_p']
    PG8 =  p['F'] - p['C_p']

    EGF = z*(y*PF1 + (1-y)*PF3) + (1-z)*(y*PF2 + (1-y)*PF4)
    ENF = z*(y*PF5 + (1-y)*PF7) + (1-z)*(y*PF6 + (1-y)*PF8)

    EAC = x*(z*PC1 + (1-z)*PC2) + (1-x)*(z*PC5 + (1-z)*PC6)
    ERC = x*(z*PC3 + (1-z)*PC4) + (1-x)*(z*PC7 + (1-z)*PC8)

    EST = x*(y*PG1 + (1-y)*PG3) + (1-x)*(y*PG5 + (1-y)*PG7)
    ESL = x*(y*PG2 + (1-y)*PG4) + (1-x)*(y*PG6 + (1-y)*PG8)

    return EGF - ENF, EAC - ERC, EST - ESL


# =============================================================================
# 2.  REPLICATOR DYNAMICS
# =============================================================================
DAMP    = 0.03
T_MAX   = 7.0
N_EVAL  = 2000
CONV_THRESH = 0.01


def replicator_ode(p):
    def ode(t, s):
        x, y, z = np.clip(s, 1e-6, 1 - 1e-6)
        try:
            dfx, dfy, dfz = calc_fitness_differentials(p, x, y, z)
            dx = DAMP * x * (1-x) * dfx
            dy = DAMP * y * (1-y) * dfy
            dz = DAMP * z * (1-z) * dfz
            out = [dx, dy, dz]
            return out if np.isfinite(out).all() else [0., 0., 0.]
        except Exception:
            return [0., 0., 0.]
    return ode


def run_one_simulation(p, target_eq, seed):
    local_rng = np.random.default_rng(seed)
    x0 = local_rng.uniform(0.1, 0.9, 3)
    ode = replicator_ode(p)
    target = np.array(target_eq, dtype=float)

    def reached_target(t, s):
        return np.linalg.norm(np.clip(s, 0, 1) - target) - CONV_THRESH
    reached_target.terminal = True
    reached_target.direction = -1

    STALL_SPEED = 1e-7
    def stalled(t, s):
        dfx, dfy, dfz = calc_fitness_differentials(p, *np.clip(s, 1e-6, 1 - 1e-6))
        x, y, z = np.clip(s, 1e-6, 1 - 1e-6)
        speed = abs(DAMP * x*(1-x)*dfx) + abs(DAMP * y*(1-y)*dfy) + abs(DAMP * z*(1-z)*dfz)
        return speed - STALL_SPEED
    stalled.terminal = True
    stalled.direction = -1

    t_eval = np.linspace(0, T_MAX, N_EVAL)
    try:
        sol = solve_ivp(ode, (0, T_MAX), x0, method='RK45',
                        t_eval=t_eval, rtol=1e-9, atol=1e-12,
                        events=[reached_target, stalled])
        if not sol.success:
            return False, x0, T_MAX
        final = sol.y[:, -1]
        if len(sol.t_events[0]) > 0:
            return True, final, float(sol.t_events[0][0])
        return False, final, float(sol.t[-1])
    except Exception:
        return False, x0, T_MAX


def evaluate_robustness(p_vec, target_eq, n_seeds=5, base_seed=0):
    p = vec_to_dict(p_vec)
    n_conv = 0
    dists  = []
    for k in range(n_seeds):
        conv, final, _ = run_one_simulation(p, target_eq, seed=base_seed*100+k)
        n_conv += int(conv)
        dists.append(np.linalg.norm(final - np.array(target_eq)))
    return n_conv, float(np.mean(dists)), n_conv / n_seeds


def fitness_for_optimizer(p_vec, target_eq, n_seeds=5, base_seed=0):
    p_vec = np.clip(p_vec, LOWER, UPPER)
    _, _, rate = evaluate_robustness(p_vec, target_eq, n_seeds, base_seed)
    return rate


# =============================================================================
# 3.  EIGENVALUE / STABILITY (post-hoc)
# =============================================================================
def analyze_stability(p_dict, eq_label):
    p = p_dict
    eigens = []

    if eq_label == 'E3':   # (x=0, y=1, z=0)
        dfx, _,   _   = calc_fitness_differentials(p, 0.001, 1.0, 0.0)
        _,   dfy, _   = calc_fitness_differentials(p, 0.0,   0.999,0.0)
        _,   _,   dfz = calc_fitness_differentials(p, 0.0,   1.0, 0.001)
        eigens = [dfx, -dfy, dfz]
        stable = (dfx <= 0) and (dfy >= 0) and (dfz <= 0)
    elif eq_label == 'E4': # (x=1, y=1, z=0)
        dfx, _,   _   = calc_fitness_differentials(p, 0.999, 1.0, 0.0)
        _,   dfy, _   = calc_fitness_differentials(p, 1.0,   0.999,0.0)
        _,   _,   dfz = calc_fitness_differentials(p, 1.0,   1.0, 0.001)
        eigens = [-dfx, -dfy, dfz]
        stable = (dfx >= 0) and (dfy >= 0) and (dfz <= 0)
    elif eq_label == 'E7': # (x=0, y=1, z=1)
        dfx, _,   _   = calc_fitness_differentials(p, 0.001, 1.0, 1.0)
        _,   dfy, _   = calc_fitness_differentials(p, 0.0,   0.999,1.0)
        _,   _,   dfz = calc_fitness_differentials(p, 0.0,   1.0, 0.999)
        eigens = [dfx, -dfy, -dfz]
        stable = (dfx <= 0) and (dfy >= 0) and (dfz >= 0)
    elif eq_label == 'E8': # (x=1, y=1, z=1)
        dfx, _,   _   = calc_fitness_differentials(p, 0.999, 1.0, 1.0)
        _,   dfy, _   = calc_fitness_differentials(p, 1.0,   0.999,1.0)
        _,   _,   dfz = calc_fitness_differentials(p, 1.0,   1.0, 0.999)
        eigens = [-dfx, -dfy, -dfz]
        stable = (dfx >= 0) and (dfy >= 0) and (dfz >= 0)
    else:
        raise ValueError(f"Unknown eq_label: {eq_label}")

    all_negative = all(e < 0 for e in eigens)
    all_strong   = all(e < -1 for e in eigens)
    return eigens, stable, all_negative, all_strong


# =============================================================================
# 4.  EQUILIBRIUM CONFIG
# =============================================================================
EQUILIBRIA = {
    'E3': {'label': 'E3', 'target': (0, 1, 0), 'name': '(Fake, Accept, Slack)',     'type': 'stable'},
    'E4': {'label': 'E4', 'target': (1, 1, 0), 'name': '(Genuine, Accept, Slack)',  'type': 'stable'},
    'E7': {'label': 'E7', 'target': (0, 1, 1), 'name': '(Fake, Accept, Strict)',    'type': 'unstable'},
    'E8': {'label': 'E8', 'target': (1, 1, 1), 'name': '(Genuine, Accept, Strict)', 'type': 'unstable'},
}


# =============================================================================
# 5.  VALIDATION (shared)
# =============================================================================
N_RUNS      = 100
N_FIT_SEEDS = 5
N_VAL_SEEDS = 5

FIELDNAMES = [
    'run_id', 'algorithm', 'equilibrium',
    'search_best_fitness', 'search_n_gen_used',
    *PARAM_NAMES,
    'val_conv_rate', 'val_n_conv', 'val_mean_dist', 'val_mean_conv_time',
    'stable', 'all_eigenvalues_negative', 'all_eigenvalues_strong',
    'lambda_x', 'lambda_y', 'lambda_z',
    'wall_time_s',
]


def validate_optimal_params(p_vec, target_eq, eq_label, run_id, n_val_seeds=N_VAL_SEEDS):
    p_vec = np.clip(p_vec, LOWER, UPPER)
    p_dict = vec_to_dict(p_vec)
    conv_flags, final_dists, conv_times = [], [], []
    for k in range(n_val_seeds):
        conv, final, ctime = run_one_simulation(p_dict, target_eq, seed=run_id*100000+90000+k)
        conv_flags.append(conv)
        final_dists.append(float(np.linalg.norm(final - np.array(target_eq))))
        conv_times.append(ctime)
    n_conv = sum(conv_flags)
    conv_rate = n_conv / n_val_seeds
    eigens, stable, all_neg, all_strong = analyze_stability(p_dict, eq_label)
    return {
        'conv_rate_validation': conv_rate,
        'n_conv_validation': n_conv,
        'mean_final_dist': float(np.mean(final_dists)),
        'mean_conv_time': float(np.mean([t for t, c in zip(conv_times, conv_flags) if c])) if n_conv > 0 else float(T_MAX),
        'stable': stable,
        'all_eigenvalues_negative': all_neg,
        'all_eigenvalues_strong': all_strong,
        'lambda_x': float(eigens[0]),
        'lambda_y': float(eigens[1]),
        'lambda_z': float(eigens[2]),
    }


def _summarise(df, algo_name, eq_key):
    eq_type = EQUILIBRIA[eq_key]['type']
    pct_fully_converged = round(100*(df['val_conv_rate'] == 1.0).mean(), 2)
    pct_any_convergence = round(100*(df['val_conv_rate'] >  0  ).mean(), 2)

    if eq_type == 'stable':
        identification_success = round(100*(df['val_conv_rate'] == 1.0).mean(), 2)
    else:
        identification_success = round(100*(df['val_conv_rate'] == 0.0).mean(), 2)

    return {
        'algorithm': algo_name, 'equilibrium': eq_key, 'eq_type': eq_type,
        'n_runs': len(df),
        'pct_fully_converged':  pct_fully_converged,
        'pct_any_convergence':  pct_any_convergence,
        'identification_success': identification_success,
        'mean_val_conv_rate':   round(df['val_conv_rate'].mean(), 4),
        'std_val_conv_rate':    round(df['val_conv_rate'].std(),  4),
        'pct_stable':           round(100*df['stable'].mean(), 2),
        'pct_all_eig_negative': round(100*df['all_eigenvalues_negative'].mean(), 2),
        'pct_all_eig_strong':   round(100*df['all_eigenvalues_strong'].mean(), 2),
        'mean_lambda_x': round(df['lambda_x'].mean(), 4), 'std_lambda_x': round(df['lambda_x'].std(), 4),
        'mean_lambda_y': round(df['lambda_y'].mean(), 4), 'std_lambda_y': round(df['lambda_y'].std(), 4),
        'mean_lambda_z': round(df['lambda_z'].mean(), 4), 'std_lambda_z': round(df['lambda_z'].std(), 4),
        'mean_val_dist': round(df['val_mean_dist'].mean(), 6), 'std_val_dist': round(df['val_mean_dist'].std(), 6),
        'mean_wall_time_s': round(df['wall_time_s'].mean(), 2),
        'total_wall_time_s': round(df['wall_time_s'].sum(), 2),
    }


def save_eigenvalue_distributions(algo_name, eq_key, df_runs):
    eig_csv = os.path.join(OUTPUT_DIR, f'{algo_name}_{eq_key}_eigenvalues.csv')
    df_runs[['run_id', 'algorithm', 'equilibrium', 'lambda_x', 'lambda_y', 'lambda_z',
             'all_eigenvalues_negative', 'all_eigenvalues_strong', 'stable',
             'val_conv_rate']].to_csv(eig_csv, index=False)


def run_search_experiment(algo_name, algo_fn, n_runs=N_RUNS):
    """
    Generic driver for a search-based model (GA / NSGA-II / PSO / CMA-ES).
    algo_fn(target_eq, eq_label, run_id, n_fit_seeds) -> (best_vec, best_fit, n_gen)
    Loops over ALL four equilibria, writes per-eq CSVs, returns list of summaries.
    """
    summaries = []
    for eq_key, eq_cfg in EQUILIBRIA.items():
        target_eq = eq_cfg['target']; eq_label = eq_cfg['label']
        out_csv     = os.path.join(OUTPUT_DIR, f'{algo_name}_{eq_key}_all_runs.csv')
        summary_csv = os.path.join(OUTPUT_DIR, f'{algo_name}_{eq_key}_summary.csv')

        print(f"\n{'='*70}\n  {algo_name}  |  {eq_key} {eq_cfg['name']}  target={target_eq}  ({eq_cfg['type']})\n{'='*70}")

        rows = []
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES); writer.writeheader()
            for run_id in range(1, n_runs+1):
                t0 = time.time()
                best_vec, best_fit, n_gen = algo_fn(target_eq, eq_label, run_id=run_id, n_fit_seeds=N_FIT_SEEDS)
                val = validate_optimal_params(best_vec, target_eq, eq_label, run_id, N_VAL_SEEDS)
                wall = time.time() - t0; p_dict = vec_to_dict(best_vec)
                row = {
                    'run_id': run_id, 'algorithm': algo_name, 'equilibrium': eq_key,
                    'search_best_fitness': round(best_fit, 6), 'search_n_gen_used': n_gen,
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
                print(f"  Run {run_id:3d}/{n_runs} {status}  conv={val['conv_rate_validation']:.2f}  "
                      f"allNegEig={val['all_eigenvalues_negative']}  t={wall:.1f}s")

        df = pd.DataFrame(rows)
        summary = _summarise(df, algo_name, eq_key)
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        save_eigenvalue_distributions(algo_name, eq_key, df)
        summaries.append(summary)
        print(f"  -> {summary['pct_fully_converged']}% fully converged | "
              f"{summary['pct_all_eig_negative']}% all-eig-negative")
    return summaries


def run_rl_experiment(n_runs=N_RUNS):
    """RL: load 100 discovered params per eq, push through SAME validation."""
    summaries = []
    for eq_key, eq_cfg in EQUILIBRIA.items():
        target_eq = eq_cfg['target']; eq_label = eq_cfg['label']
        csv_path = os.path.join(RL_CSV_DIR, RL_CSV_TEMPLATE.format(eq=eq_key))
        if not os.path.exists(csv_path):
            print(f"  !! RL CSV not found for {eq_key}: {csv_path} -> skipping")
            continue

        rl_df = pd.read_csv(csv_path)
        out_csv     = os.path.join(OUTPUT_DIR, f'RL_{eq_key}_all_runs.csv')
        summary_csv = os.path.join(OUTPUT_DIR, f'RL_{eq_key}_summary.csv')

        print(f"\n{'='*70}\n  RL (ours)  |  {eq_key} {eq_cfg['name']}  target={target_eq}  ({eq_cfg['type']})\n{'='*70}")
        print(f"  Loaded {len(rl_df)} optimal parameter sets from {os.path.basename(csv_path)}")

        n = min(n_runs, len(rl_df))
        rows = []
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES); writer.writeheader()
            for i in range(n):
                run_id = i + 1
                t0 = time.time()
                p_vec = np.array([float(rl_df.iloc[i][nm]) for nm in PARAM_NAMES], dtype=float)
                p_vec = np.clip(p_vec, LOWER, UPPER)
                val = validate_optimal_params(p_vec, target_eq, eq_label, run_id, N_VAL_SEEDS)
                wall = time.time() - t0; p_dict = vec_to_dict(p_vec)
                row = {
                    'run_id': run_id, 'algorithm': 'RL', 'equilibrium': eq_key,
                    'search_best_fitness': '', 'search_n_gen_used': '',
                    **{nm: round(float(p_dict[nm]), 6) for nm in PARAM_NAMES},
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
                if run_id % 20 == 0:
                    print(f"    validated {run_id}/{n}")

        df = pd.DataFrame(rows)
        summary = _summarise(df, 'RL', eq_key)
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        save_eigenvalue_distributions('RL', eq_key, df)
        summaries.append(summary)
        print(f"  -> {summary['pct_fully_converged']}% fully converged | "
              f"{summary['pct_all_eig_negative']}% all-eig-negative")
    return summaries