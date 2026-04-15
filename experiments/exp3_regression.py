"""
Experiment 3 — 1D Piecewise Regression: GA-Optimized Step Positions
====================================================================
GA evolves step neuron positions (the key non-differentiable params).
Tanh neurons have fixed random input weights (ELM philosophy).
Output weights solved analytically by ridge regression (lstsq).

Each GA individual = a network with K step neurons at positions [x1, ..., xK].
The GA searches for the NUMBER and POSITIONS of steps simultaneously.

Hypothesis: 4 well-placed steps + 80 tanh should beat 80 random steps + 120 tanh.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import json, os, time, copy

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Target function
# ═══════════════════════════════════════════════════════════════════════════════
SEGMENTS = [
    (-5, -3, -2, True),   # include x=-5 (left boundary)
    (-3, -1,  1, True),
    (-1,  1, -1, True),
    ( 1,  3,  3, True),
    ( 3,  5,  0, True),
]
X_MIN, X_MAX = -5.0, 5.0
SEED_DATA = 42
DENSITY = 20

def evaluate_step_fn(x):
    for i, (x0, x1, yv, lc) in enumerate(SEGMENTS):
        last = (i == len(SEGMENTS) - 1)
        in_l = (x >= x0) if lc else (x > x0)
        in_r = (x <= x1) if last else (x < x1)
        if in_l and in_r:
            return yv
    return np.nan

def make_spline(seed=SEED_DATA, n_knots=8, amplitude=1.5):
    rng = np.random.default_rng(seed)
    kx = np.linspace(X_MIN, X_MAX, n_knots)
    ky = rng.uniform(-amplitude, amplitude, size=n_knots)
    return CubicSpline(kx, ky, bc_type='natural')

def evaluate_combined(x_arr, cs):
    step_vals = np.array([evaluate_step_fn(x) for x in x_arr])
    return step_vals + cs(x_arr)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ELM building blocks
# ═══════════════════════════════════════════════════════════════════════════════
def tanh_hidden(x, n_tanh, seed=7, scale=2.5):
    """Random tanh hidden activations. x: (N,) → H: (N, n_tanh)"""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=(n_tanh, 1))
    b = rng.uniform(-scale * (X_MAX - X_MIN) / 2,
                     scale * (X_MAX - X_MIN) / 2, size=n_tanh)
    z = x.reshape(-1, 1) @ W.T + b
    return np.tanh(z)

def step_hidden(x, positions, kappa=100.0):
    """Sigmoid step neurons at given positions. x: (N,) → H: (N, K)"""
    if len(positions) == 0:
        return np.empty((len(x), 0))
    positions = np.asarray(positions)
    z = kappa * (x.reshape(-1, 1) - positions.reshape(1, -1))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def solve_ridge(H, y, lam=1e-6):
    """Ridge regression: beta = (H'H + λI)^{-1} H'y"""
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fitness evaluation
# ═══════════════════════════════════════════════════════════════════════════════
# Precompute tanh hidden matrix (fixed across all individuals)
_H_TANH_CACHE = {}

def get_tanh_hidden(x, n_tanh, seed=7):
    key = (id(x), n_tanh, seed)
    if key not in _H_TANH_CACHE:
        _H_TANH_CACHE[key] = tanh_hidden(x, n_tanh, seed)
    return _H_TANH_CACHE[key]

def evaluate_individual(positions, x_train, y_train, x_val, y_val,
                        n_tanh, kappa, lam, seed_tanh, parsimony):
    """
    Evaluate a GA individual (set of step positions).
    Returns fitness (lower is better) = validation RMSE + parsimony * n_steps.
    """
    H_tanh = get_tanh_hidden(x_train, n_tanh, seed_tanh)
    H_step = step_hidden(x_train, positions, kappa)
    H = np.hstack([H_tanh, H_step])
    beta = solve_ridge(H, y_train, lam)

    # Evaluate on validation set
    H_tanh_v = get_tanh_hidden(x_val, n_tanh, seed_tanh)
    H_step_v = step_hidden(x_val, positions, kappa)
    H_v = np.hstack([H_tanh_v, H_step_v])
    y_pred = H_v @ beta

    val_rmse = rmse(y_val, y_pred)
    fitness = val_rmse + parsimony * len(positions)
    return fitness, val_rmse, beta

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════════
class Individual:
    """GA individual: a list of step positions."""
    def __init__(self, positions):
        self.positions = sorted(positions)
        self.fitness = np.inf
        self.val_rmse = np.inf
        self.beta = None

    def copy(self):
        ind = Individual(list(self.positions))
        ind.fitness = self.fitness
        ind.val_rmse = self.val_rmse
        ind.beta = self.beta
        return ind

def init_population(pop_size, max_steps, rng):
    """Initialize population with variable number of steps."""
    pop = []
    for _ in range(pop_size):
        n_steps = rng.integers(1, max_steps + 1)
        positions = sorted(rng.uniform(X_MIN + 0.5, X_MAX - 0.5, size=n_steps).tolist())
        pop.append(Individual(positions))
    return pop

def tournament_select(pop, k=3, rng=None):
    """Tournament selection (k individuals, pick best)."""
    indices = rng.choice(len(pop), size=k, replace=False)
    best = min(indices, key=lambda i: pop[i].fitness)
    return pop[best].copy()

def crossover(p1, p2, rng):
    """Combine step positions from two parents."""
    all_pos = sorted(set(p1.positions + p2.positions))
    if len(all_pos) == 0:
        return Individual([])
    # Random subset
    n = rng.integers(1, min(len(all_pos), 8) + 1)
    chosen = sorted(rng.choice(all_pos, size=n, replace=False).tolist())
    return Individual(chosen)

def mutate(ind, rng, mutation_rate=0.3, max_steps=8):
    """Mutate an individual: shift, add, or remove a step."""
    positions = list(ind.positions)

    for i in range(len(positions)):
        if rng.random() < mutation_rate:
            # Shift position by small amount
            positions[i] += rng.normal(0, 0.3)
            positions[i] = np.clip(positions[i], X_MIN + 0.1, X_MAX - 0.1)

    # Add a step (10% chance)
    if rng.random() < 0.10 and len(positions) < max_steps:
        new_pos = rng.uniform(X_MIN + 0.5, X_MAX - 0.5)
        positions.append(new_pos)

    # Remove a step (10% chance, if more than 1)
    if rng.random() < 0.10 and len(positions) > 1:
        idx = rng.integers(0, len(positions))
        positions.pop(idx)

    return Individual(sorted(positions))

def merge_close_steps(positions, min_dist=0.3):
    """Merge step neurons that are too close together."""
    if len(positions) <= 1:
        return positions
    merged = [positions[0]]
    for p in positions[1:]:
        if abs(p - merged[-1]) < min_dist:
            merged[-1] = (merged[-1] + p) / 2  # average
        else:
            merged.append(p)
    return merged

def run_ga(x_train, y_train, x_val, y_val, config):
    """Run the full GA optimization."""
    rng = np.random.default_rng(config['seed_ga'])

    pop = init_population(config['pop_size'], config['max_steps'], rng)

    best_ever = None
    history = []

    for gen in range(config['n_gen']):
        # Evaluate population
        for ind in pop:
            ind.fitness, ind.val_rmse, ind.beta = evaluate_individual(
                ind.positions, x_train, y_train, x_val, y_val,
                config['n_tanh'], config['kappa'], config['lam'],
                config['seed_tanh'], config['parsimony'])

        # Sort by fitness
        pop.sort(key=lambda ind: ind.fitness)

        # Track best
        if best_ever is None or pop[0].fitness < best_ever.fitness:
            best_ever = pop[0].copy()

        history.append({
            'gen': gen,
            'best_fitness': float(pop[0].fitness),
            'best_val_rmse': float(pop[0].val_rmse),
            'best_n_steps': len(pop[0].positions),
            'best_positions': list(pop[0].positions),
            'mean_fitness': float(np.mean([ind.fitness for ind in pop])),
        })

        if gen % 5 == 0 or gen == config['n_gen'] - 1:
            pos_str = ', '.join(f'{p:.2f}' for p in pop[0].positions)
            print(f"  Gen {gen:3d}  |  best RMSE={pop[0].val_rmse:.6f}  "
                  f"steps={len(pop[0].positions)} [{pos_str}]  "
                  f"|  mean={history[-1]['mean_fitness']:.4f}")

        # Elitism: keep top elite_count
        elite = config['elite_count']
        new_pop = [ind.copy() for ind in pop[:elite]]

        # Fill rest with offspring
        while len(new_pop) < config['pop_size']:
            p1 = tournament_select(pop, k=3, rng=rng)
            p2 = tournament_select(pop, k=3, rng=rng)
            child = crossover(p1, p2, rng)
            child = mutate(child, rng, config['mutation_rate'], config['max_steps'])
            child.positions = merge_close_steps(child.positions, min_dist=0.2)
            new_pop.append(child)

        pop = new_pop

    # ── Local refinement with Nelder-Mead on the best ────────────────────────
    if config.get('nm_refine', True) and len(best_ever.positions) > 0:
        print(f"\n  Nelder-Mead refinement on best ({len(best_ever.positions)} steps)...")

        def nm_objective(pos_vec):
            positions = sorted(pos_vec.tolist())
            f, _, _ = evaluate_individual(
                positions, x_train, y_train, x_val, y_val,
                config['n_tanh'], config['kappa'], config['lam'],
                config['seed_tanh'], 0.0)  # no parsimony during NM
            return f

        res = minimize(nm_objective, np.array(best_ever.positions),
                       method='Nelder-Mead',
                       options={'maxfev': config.get('nm_maxfev', 200),
                                'xatol': 1e-4, 'fatol': 1e-6})

        refined_pos = sorted(res.x.tolist())
        f_refined, rmse_refined, beta_refined = evaluate_individual(
            refined_pos, x_train, y_train, x_val, y_val,
            config['n_tanh'], config['kappa'], config['lam'],
            config['seed_tanh'], config['parsimony'])

        print(f"  Before NM: RMSE={best_ever.val_rmse:.6f}  pos={[f'{p:.3f}' for p in best_ever.positions]}")
        print(f"  After NM:  RMSE={rmse_refined:.6f}  pos={[f'{p:.3f}' for p in refined_pos]}")

        if f_refined < best_ever.fitness:
            best_ever.positions = refined_pos
            best_ever.fitness = f_refined
            best_ever.val_rmse = rmse_refined
            best_ever.beta = beta_refined

    return best_ever, history

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    # Generate data
    cs = make_spline()
    n_train = max(2, int(round(DENSITY * (X_MAX - X_MIN))))
    x_train = np.linspace(X_MIN, X_MAX, n_train)
    y_train = evaluate_combined(x_train, cs)

    # Validation: dense grid (the "real" function)
    x_val = np.linspace(X_MIN, X_MAX, 2000)
    y_val = evaluate_combined(x_val, cs)

    print(f"Training: {n_train} pts  |  Validation: {len(x_val)} pts")
    print("=" * 90)

    # ── GA Configuration ─────────────────────────────────────────────────────
    config = {
        'n_tanh': 80,
        'kappa': 500.0,       # sharper steps (oracle showed kappa=500 is best)
        'lam': 1e-6,
        'seed_tanh': 7,
        'seed_ga': 42,
        'pop_size': 50,       # larger population for better coverage
        'n_gen': 80,          # more generations
        'elite_count': 5,
        'mutation_rate': 0.3,
        'max_steps': 8,
        'parsimony': 0.002,   # slightly stronger parsimony
        'nm_refine': True,
        'nm_maxfev': 500,
    }

    print(f"GA config: pop={config['pop_size']}, gen={config['n_gen']}, "
          f"max_steps={config['max_steps']}, kappa={config['kappa']}, "
          f"tanh={config['n_tanh']}, parsimony={config['parsimony']}")
    print("-" * 90)

    # ── Run GA ───────────────────────────────────────────────────────────────
    best, history = run_ga(x_train, y_train, x_val, y_val, config)

    elapsed = time.time() - t_start
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"  Steps found:     {len(best.positions)}")
    print(f"  Positions:       {[f'{p:.4f}' for p in best.positions]}")
    print(f"  True positions:  [-3.0, -1.0, 1.0, 3.0]")
    print(f"  Position errors: {[f'{abs(p - t):.4f}' for p, t in zip(best.positions, [-3, -1, 1, 3])] if len(best.positions) == 4 else 'N/A (different count)'}")
    print(f"  Validation RMSE: {best.val_rmse:.6f}")
    print(f"  Total neurons:   {config['n_tanh']} tanh + {len(best.positions)} step = {config['n_tanh'] + len(best.positions)}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"\n  [REF] Prof iterative hybrid:  RMSE=0.162196  (200 neurons)")
    print(f"  [REF] Oracle (4 steps):       see baseline experiment")

    # Step output weights
    n_tanh = config['n_tanh']
    step_weights = best.beta[n_tanh:].tolist() if best.beta is not None else []
    print(f"  Step output weights: {[f'{w:.4f}' for w in step_weights]}")

    # ── Dense prediction for plotting ────────────────────────────────────────
    H_tanh_d = get_tanh_hidden(x_val, config['n_tanh'], config['seed_tanh'])
    H_step_d = step_hidden(x_val, best.positions, config['kappa'])
    H_d = np.hstack([H_tanh_d, H_step_d])
    y_pred = H_d @ best.beta

    # Also compute train RMSE
    H_tanh_t = get_tanh_hidden(x_train, config['n_tanh'], config['seed_tanh'])
    H_step_t = step_hidden(x_train, best.positions, config['kappa'])
    H_t = np.hstack([H_tanh_t, H_step_t])
    y_pred_train = H_t @ best.beta
    train_rmse = rmse(y_train, y_pred_train)
    print(f"  Train RMSE:      {train_rmse:.6f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Save results
    # ═══════════════════════════════════════════════════════════════════════════
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    result = {
        'config': config,
        'best_positions': best.positions,
        'best_val_rmse': float(best.val_rmse),
        'best_train_rmse': float(train_rmse),
        'best_fitness': float(best.fitness),
        'n_steps_found': len(best.positions),
        'step_weights': step_weights,
        'true_positions': [-3.0, -1.0, 1.0, 3.0],
        'elapsed_s': elapsed,
        'history': history,
    }

    with open(os.path.join(results_dir, 'ga_results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Plots
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False,
                             gridspec_kw={'hspace': 0.25,
                                          'height_ratios': [3, 2, 1.5, 1.5]})
    fig.patch.set_facecolor('#0d0d0f')

    teal = '#5ac8c0'
    coral = '#e07060'
    gold = '#c8a85a'
    cyan = '#5ab4c8'
    green = '#7acc7a'
    purple = '#b07acc'

    for ax in axes:
        ax.set_facecolor('#12121a')
        ax.grid(True, color='#1e1e2a', linewidth=0.8, zorder=0)
        ax.tick_params(colors='#4a4a60', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a38')

    # ── Panel 0: Prediction vs truth ─────────────────────────────────────────
    ax = axes[0]
    ax.axhline(0, color='#3a3a50', linewidth=1.2, zorder=1)
    ax.plot(x_val, y_val, color=teal, linewidth=2.5, zorder=3,
            label='Target h(x)')
    ax.plot(x_val, y_pred, color=green, linewidth=2, linestyle='--', zorder=4,
            label=f'GA-ELM ({len(best.positions)} steps, RMSE={best.val_rmse:.4f})')
    ax.scatter(x_train, y_train, color=cyan, s=15, zorder=5,
               edgecolors='#0d0d0f', linewidths=0.4, alpha=0.6)
    # Mark found step positions
    for i, xp in enumerate(best.positions):
        lbl = f'Step at x={xp:.2f}' if i == 0 else None
        ax.axvline(xp, color=gold, linewidth=1.5, linestyle='--', alpha=0.7,
                   zorder=2, label=lbl)
    # Mark true positions
    for xp in [-3, -1, 1, 3]:
        ax.axvline(xp, color=purple, linewidth=1, linestyle=':', alpha=0.4, zorder=2)
    ax.set_ylabel('h(x)', color='#6b6659', fontsize=10)
    ax.set_title(f'GA-ELM: {len(best.positions)} Interpretable Steps Found '
                 f'(RMSE={best.val_rmse:.4f} vs Prof 0.1622)',
                 color='#e8e2d5', fontsize=13, fontfamily='serif', pad=10)
    ax.set_ylim(-6, 7)
    _legend(ax)

    info = (f"tanh={config['n_tanh']}, kappa={config['kappa']}\n"
            f"GA: pop={config['pop_size']}, gen={config['n_gen']}\n"
            f"parsimony={config['parsimony']}\n"
            f"positions: {[f'{p:.2f}' for p in best.positions]}\n"
            f"true:      [-3, -1, 1, 3]")
    ax.text(0.99, 0.97, info, transform=ax.transAxes,
            fontsize=8, va='top', ha='right', fontfamily='monospace',
            color='#e8e2d5',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#1a1a24',
                      edgecolor='#2a2a38', alpha=0.9))

    # ── Panel 1: Step neuron contributions ───────────────────────────────────
    ax = axes[1]
    ax.axhline(0, color='#3a3a50', linewidth=1.2, zorder=1)
    colors_step = ['#e07060', '#c8a85a', '#7acc7a', '#5ab4c8',
                   '#b07acc', '#cc7a7a', '#7accb0', '#acc87a']
    total_step_contrib = np.zeros(len(x_val))
    for i, xp in enumerate(best.positions):
        w = best.beta[n_tanh + i]
        h_i = step_hidden(x_val, [xp], config['kappa']).ravel() * w
        total_step_contrib += h_i
        c = colors_step[i % len(colors_step)]
        ax.plot(x_val, h_i, color=c, linewidth=1.5, alpha=0.7,
                label=f'Step {i+1}: x={xp:.2f}, w={w:.2f}')
    ax.plot(x_val, total_step_contrib, color=gold, linewidth=2.2, zorder=3,
            label='Total step contribution')
    ax.set_ylabel('step contrib.', color='#6b6659', fontsize=10)
    ax.set_title('Individual Step Neuron Contributions',
                 color='#e8e2d5', fontsize=12, fontfamily='serif', pad=8)
    _legend(ax)

    # ── Panel 2: Residual ────────────────────────────────────────────────────
    ax = axes[2]
    residual = y_val - y_pred
    ax.fill_between(x_val, residual, 0, color=(0.87, 0.44, 0.38, 0.15), zorder=2)
    ax.plot(x_val, residual, color=coral, linewidth=1.5, zorder=3,
            label=f'Residual (max={np.max(np.abs(residual)):.3f})')
    ax.set_ylabel('residual', color='#6b6659', fontsize=10)
    ax.set_xlabel('x', color='#6b6659', fontsize=11)
    _legend(ax)

    # ── Panel 3: GA convergence ──────────────────────────────────────────────
    ax = axes[3]
    gens = [h['gen'] for h in history]
    best_rmses = [h['best_val_rmse'] for h in history]
    mean_fits = [h['mean_fitness'] for h in history]
    ax.plot(gens, best_rmses, color=green, linewidth=2, label='Best val RMSE')
    ax.plot(gens, mean_fits, color=coral, linewidth=1.5, alpha=0.6,
            label='Mean fitness')
    ax.axhline(0.1622, color=purple, linewidth=1, linestyle='--', alpha=0.5,
               label='Prof ref (0.1622)')
    ax.set_xlabel('Generation', color='#6b6659', fontsize=11)
    ax.set_ylabel('RMSE / Fitness', color='#6b6659', fontsize=10)
    ax.set_title('GA Convergence', color='#e8e2d5', fontsize=12,
                 fontfamily='serif', pad=8)
    _legend(ax)

    fname = os.path.join(results_dir, 'exp3_ga_results.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")
    plt.show()


def _legend(ax):
    leg = ax.legend(loc='upper left', fontsize=8,
                    facecolor='#1a1a24', edgecolor='#2a2a38', framealpha=0.9)
    for t in leg.get_texts():
        t.set_color('#e8e2d5')


if __name__ == '__main__':
    main()
