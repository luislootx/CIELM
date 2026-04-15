"""
Experiment 10 — ELM-CINN for Linear Acoustics (PDE System)
=========================================================
Replicates CINN paper (arXiv 2212.14012) Section 4.2 and extends
with Riemann IC (professor's request).

System:
  ∂p/∂t + K₀ ∂v/∂x = 0
  ∂v/∂t + (c₀²/K₀) ∂p/∂x = 0

Diagonalization (A = R Λ R⁻¹):
  w = R⁻¹ u  →  w¹ travels LEFT at c₀,  w² travels RIGHT at c₀
  w¹(x,t) = w¹₀(x + c₀t)    (left-going wave)
  w²(x,t) = w²₀(x - c₀t)    (right-going wave)

Reconstruction:
  p(x,t) = Z₀ (w² - w¹)     v(x,t) = w¹ + w²
  where Z₀ = K₀/c₀ (impedance)

ELM-CINN approach:
  Fit w¹(ξ₁) and w²(ξ₂) with separate ELMs.
  Evaluate at ξ₁ = x + c₀t  and  ξ₂ = x - c₀t.
  Reconstruct p and v via R.

Sub-experiments:
  10A: Gaussian IC (CINN paper setup): p₀ = exp(-100x²), v₀ = 0
  10B: Riemann IC: p₀ = p_L (x<0), p_R (x>0), v₀ = 0
       → two step neurons track the left and right waves

Reference (CINN paper Table 4, hidden acoustics, 10 reps):
  CINN 1000 iter: pressure L2=0.1267, velocity L2=0.0588, time=9.1s
  PINN 5000 iter: pressure L2=0.5209, velocity L2=0.0615, time=44.4s
"""

import numpy as np
import matplotlib.pyplot as plt
import json, os, time

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════════════════════════════════════
C0 = 1.0    # sound speed
Z0 = 1.0    # impedance K₀/c₀
K0 = Z0 * C0

# Domain large enough for waves to stay inside
X_MIN, X_MAX = -1.5, 1.5
T_FINAL = 0.8

# ── 10A: Gaussian IC (CINN paper) ──
def gaussian_ic_p(x):
    return np.exp(-100 * x**2)

def gaussian_ic_v(x):
    return np.zeros_like(x)

def gaussian_exact_p(x, t):
    return 0.5 * (np.exp(-100*(x - C0*t)**2) + np.exp(-100*(x + C0*t)**2))

def gaussian_exact_v(x, t):
    return 1.0/(2*Z0) * (np.exp(-100*(x - C0*t)**2) - np.exp(-100*(x + C0*t)**2))

# ── 10B: Riemann IC ──
P_L, P_R = 2.0, 0.0

def riemann_ic_p(x):
    return np.where(x < 0, P_L, P_R)

def riemann_ic_v(x):
    return np.zeros_like(x)

def riemann_exact_p(x, t):
    """p = Z₀(w² - w¹), w²₀ = p₀/(2Z₀), w¹₀ = -p₀/(2Z₀)."""
    w2 = riemann_ic_p(x - C0*t) / (2*Z0)   # right-going
    w1 = -riemann_ic_p(x + C0*t) / (2*Z0)  # left-going
    return Z0 * (w2 - w1)

def riemann_exact_v(x, t):
    w2 = riemann_ic_p(x - C0*t) / (2*Z0)
    w1 = -riemann_ic_p(x + C0*t) / (2*Z0)
    return w1 + w2

# ── Characteristic decomposition ──
def ic_to_char(p0, v0):
    """Convert (p,v) IC to characteristic variables (w1, w2)."""
    w1 = (-p0/Z0 + v0) / 2.0   # left-going
    w2 = (p0/Z0 + v0) / 2.0    # right-going
    return w1, w2

def char_to_pv(w1, w2):
    """Convert characteristic variables back to (p, v)."""
    p = Z0 * (w2 - w1)
    v = w1 + w2
    return p, v


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Network building blocks
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tanh_weights(n_tanh, seed=7, scale=2.5):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale * (X_MAX - X_MIN), scale * (X_MAX - X_MIN),
                     size=n_tanh)
    return W, b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def hidden_matrix(x, W_tanh, b_tanh, positions, kappa):
    z_tanh = np.outer(x, W_tanh) + b_tanh
    H_tanh = np.tanh(z_tanh)
    if len(positions) > 0:
        z_step = kappa * (x.reshape(-1, 1) - np.array(positions).reshape(1, -1))
        H_step = sigmoid(z_step)
        return np.hstack([H_tanh, H_step])
    return H_tanh

def solve_ridge(H, y, lam=1e-6):
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELM-CINN solver for acoustic system
# ═══════════════════════════════════════════════════════════════════════════════
def cielm_acoustics(config, ic_p, ic_v, exact_p, exact_v, snap_times):
    """
    ELM-CINN for linear acoustics.

    1. Decompose IC into characteristic variables w1, w2
    2. Fit w1(ξ) and w2(ξ) with separate ELMs
    3. At time t: evaluate w1 at ξ₁ = x + c₀t, w2 at ξ₂ = x - c₀t
    4. Reconstruct p, v from w1, w2
    """
    n_tanh = config['n_tanh']
    kappa = config['kappa']
    seed_tanh = config['seed_tanh']
    positions = np.array(config.get('positions', []))
    K = len(positions)

    # Extended fitting domain (waves travel at c₀, need margin for t up to T)
    margin = C0 * T_FINAL + 0.3
    xi_min = X_MIN - margin
    xi_max = X_MAX + margin

    # Generate two sets of tanh weights (different seeds for w1 and w2)
    W_tanh1, b_tanh1 = generate_tanh_weights(n_tanh, seed_tanh)
    W_tanh2, b_tanh2 = generate_tanh_weights(n_tanh, seed_tanh + 100)

    # Sample IC on extended domain
    x_ic = np.linspace(xi_min, xi_max, config['n_ic'])
    p0 = ic_p(x_ic)
    v0 = ic_v(x_ic)
    w1_0, w2_0 = ic_to_char(p0, v0)

    # Fit w1 and w2 separately
    H1 = hidden_matrix(x_ic, W_tanh1, b_tanh1, positions, kappa)
    H2 = hidden_matrix(x_ic, W_tanh2, b_tanh2, positions, kappa)
    beta1 = solve_ridge(H1, w1_0, config['lam'])
    beta2 = solve_ridge(H2, w2_0, config['lam'])

    ic_rmse_w1 = float(np.sqrt(np.mean((H1 @ beta1 - w1_0)**2)))
    ic_rmse_w2 = float(np.sqrt(np.mean((H2 @ beta2 - w2_0)**2)))

    # Evaluate at each snapshot
    x_eval = np.linspace(X_MIN, X_MAX, config['n_eval'])
    snapshots = {}

    t_start = time.time()
    for t_snap in snap_times:
        xi1 = x_eval + C0 * t_snap   # left-going: ξ₁ = x + c₀t
        xi2 = x_eval - C0 * t_snap   # right-going: ξ₂ = x - c₀t

        H1_t = hidden_matrix(xi1, W_tanh1, b_tanh1, positions, kappa)
        H2_t = hidden_matrix(xi2, W_tanh2, b_tanh2, positions, kappa)
        w1_pred = H1_t @ beta1
        w2_pred = H2_t @ beta2

        p_pred, v_pred = char_to_pv(w1_pred, w2_pred)
        p_ref = exact_p(x_eval, t_snap)
        v_ref = exact_v(x_eval, t_snap)

        # Use max of (norm_ref, norm_pred, 1) to avoid division by ~0
        # when the reference field is near-zero (e.g., v(x,0)=0)
        norm_p = max(np.linalg.norm(p_ref), np.linalg.norm(p_pred), 1.0)
        norm_v = max(np.linalg.norm(v_ref), np.linalg.norm(v_pred), 1.0)

        snapshots[f"t={t_snap:.2f}"] = {
            't': float(t_snap),
            'p_pred': p_pred, 'v_pred': v_pred,
            'p_ref': p_ref, 'v_ref': v_ref,
            'p_l2': float(np.linalg.norm(p_pred - p_ref) / norm_p),
            'v_l2': float(np.linalg.norm(v_pred - v_ref) / norm_v),
            'p_rmse': float(np.sqrt(np.mean((p_pred - p_ref)**2))),
            'v_rmse': float(np.sqrt(np.mean((v_pred - v_ref)**2))),
        }
    elapsed = time.time() - t_start

    return {
        'ic_rmse_w1': ic_rmse_w1,
        'ic_rmse_w2': ic_rmse_w2,
        'elapsed_s': elapsed,
        'snapshots': snapshots,
        'x_eval': x_eval,
        'n_params': 2 * (n_tanh + K),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════════════════════════
C_BLUE   = '#2166ac'
C_RED    = '#d6604d'
C_GREEN  = '#1b7837'
C_ORANGE = '#e08214'
C_TEXT   = '#333333'
BG       = '#ffffff'
BG_AX    = '#fafafa'
GRID     = '#e0e0e0'
SPINE    = '#aaaaaa'

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)

def add_legend(ax, **kw):
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=SPINE, **kw)

def plot_acoustics_snapshots(x_eval, snapshots, title, fname):
    """Plot p and v at multiple time snapshots."""
    snap_keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(snap_keys)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), sharex=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, key in enumerate(snap_keys):
        s = snapshots[key]

        # Pressure
        ax = axes[0, i]
        style_ax(ax)
        ax.plot(x_eval, s['p_ref'], color=C_BLUE, linewidth=2.5, zorder=3,
                label='Exact')
        ax.plot(x_eval, s['p_pred'], color=C_RED, linewidth=2, linestyle='--',
                zorder=4, label='ELM-CINN')
        ax.set_title(f"t={s['t']:.2f}  p L$_2$={s['p_l2']:.4f}",
                     color=C_TEXT, fontsize=10, fontfamily='serif')
        if i == 0:
            ax.set_ylabel('p(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')

        # Velocity
        ax = axes[1, i]
        style_ax(ax)
        ax.plot(x_eval, s['v_ref'], color=C_BLUE, linewidth=2.5, zorder=3,
                label='Exact')
        ax.plot(x_eval, s['v_pred'], color=C_RED, linewidth=2, linestyle='--',
                zorder=4, label='ELM-CINN')
        ax.set_title(f"v L$_2$={s['v_l2']:.4f}",
                     color=C_TEXT, fontsize=10, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('v(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')

    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════
def run_exp10a(results_dir):
    """Exp 10A: Gaussian IC (CINN paper setup)."""
    print(f"\n{'='*70}")
    print(f"  Exp 10A: Linear Acoustics, Gaussian IC")
    print(f"{'='*70}")

    config = {
        'n_tanh': 200, 'kappa': 500.0, 'lam': 1e-8, 'seed_tanh': 7,
        'positions': [],  # smooth IC, no steps
        'n_ic': 1000, 'n_eval': 1000,
    }

    snap_times = [0.0, 0.05, 0.40, 0.81]

    # 10-seed statistics
    all_p_l2, all_v_l2, all_time = [], [], []
    for seed in range(10):
        cfg = {**config, 'seed_tanh': seed}
        res = cielm_acoustics(cfg, gaussian_ic_p, gaussian_ic_v,
                               gaussian_exact_p, gaussian_exact_v,
                               [0.0, T_FINAL])
        final = res['snapshots'][f"t={T_FINAL:.2f}"]
        all_p_l2.append(final['p_l2'])
        all_v_l2.append(final['v_l2'])
        all_time.append(res['elapsed_s'])

    print(f"  ELM-CINN (10 seeds):")
    print(f"    p L2 = {np.mean(all_p_l2):.4f}+-{np.std(all_p_l2):.4f}")
    print(f"    v L2 = {np.mean(all_v_l2):.4f}+-{np.std(all_v_l2):.4f}")
    print(f"    time = {np.mean(all_time):.3f}s")

    # Single-seed snapshots for visualization
    res_show = cielm_acoustics(config, gaussian_ic_p, gaussian_ic_v,
                                gaussian_exact_p, gaussian_exact_v,
                                snap_times)
    print(f"  IC fit: w1 RMSE={res_show['ic_rmse_w1']:.6f}, "
          f"w2 RMSE={res_show['ic_rmse_w2']:.6f}")
    for key, snap in sorted(res_show['snapshots'].items()):
        print(f"    {key}: p_L2={snap['p_l2']:.4f}, v_L2={snap['v_l2']:.4f}")

    plot_acoustics_snapshots(
        res_show['x_eval'], res_show['snapshots'],
        "Exp 10A: ELM-CINN — Linear Acoustics, Gaussian IC",
        os.path.join(results_dir, 'exp10a_gaussian_snapshots.png'))

    return {
        'p_l2_mean': float(np.mean(all_p_l2)),
        'p_l2_std': float(np.std(all_p_l2)),
        'v_l2_mean': float(np.mean(all_v_l2)),
        'v_l2_std': float(np.std(all_v_l2)),
        'time_mean': float(np.mean(all_time)),
        'all_p_l2': all_p_l2, 'all_v_l2': all_v_l2,
    }


def run_exp10b(results_dir):
    """Exp 10B: Riemann IC (step neurons for acoustic waves)."""
    print(f"\n{'='*70}")
    print(f"  Exp 10B: Linear Acoustics, Riemann IC (step neurons)")
    print(f"{'='*70}")

    config_steps = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6, 'seed_tanh': 7,
        'positions': [0.0],  # 1 step at x=0 (discontinuity)
        'n_ic': 500, 'n_eval': 1000,
    }
    config_no_steps = {**config_steps, 'positions': []}

    snap_times = [0.0, 0.20, 0.50, 0.80]

    # With step neurons
    all_p_s, all_v_s = [], []
    for seed in range(10):
        cfg = {**config_steps, 'seed_tanh': seed}
        res = cielm_acoustics(cfg, riemann_ic_p, riemann_ic_v,
                               riemann_exact_p, riemann_exact_v,
                               [0.0, T_FINAL])
        final = res['snapshots'][f"t={T_FINAL:.2f}"]
        all_p_s.append(final['p_l2'])
        all_v_s.append(final['v_l2'])

    # Without step neurons
    all_p_n, all_v_n = [], []
    for seed in range(10):
        cfg = {**config_no_steps, 'seed_tanh': seed}
        res = cielm_acoustics(cfg, riemann_ic_p, riemann_ic_v,
                               riemann_exact_p, riemann_exact_v,
                               [0.0, T_FINAL])
        final = res['snapshots'][f"t={T_FINAL:.2f}"]
        all_p_n.append(final['p_l2'])
        all_v_n.append(final['v_l2'])

    print(f"  With step neurons (10 seeds):")
    print(f"    p L2 = {np.mean(all_p_s):.4f}+-{np.std(all_p_s):.4f}")
    print(f"    v L2 = {np.mean(all_v_s):.4f}+-{np.std(all_v_s):.4f}")
    print(f"  Without step neurons:")
    print(f"    p L2 = {np.mean(all_p_n):.4f}+-{np.std(all_p_n):.4f}")
    print(f"    v L2 = {np.mean(all_v_n):.4f}+-{np.std(all_v_n):.4f}")

    # Snapshots with steps
    res_show = cielm_acoustics(config_steps, riemann_ic_p, riemann_ic_v,
                                riemann_exact_p, riemann_exact_v,
                                snap_times)
    for key, snap in sorted(res_show['snapshots'].items()):
        print(f"    {key}: p_L2={snap['p_l2']:.4f}, v_L2={snap['v_l2']:.4f}")

    plot_acoustics_snapshots(
        res_show['x_eval'], res_show['snapshots'],
        "Exp 10B: ELM-CINN — Linear Acoustics, Riemann IC",
        os.path.join(results_dir, 'exp10b_riemann_snapshots.png'))

    # Ablation bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    labels = ['ELM-CINN\n+ steps', 'ELM-CINN\ntanh only']
    for ax_idx, (field, vals_s, vals_n, ylabel) in enumerate([
        ('pressure', all_p_s, all_p_n, 'p(x,t)'),
        ('velocity', all_v_s, all_v_n, 'v(x,t)')
    ]):
        ax = axes[ax_idx]
        style_ax(ax)
        means = [np.mean(vals_s), np.mean(vals_n)]
        stds = [np.std(vals_s), np.std(vals_n)]
        colors = [C_GREEN, C_ORANGE]
        ax.bar([0, 1], means, yerr=stds, capsize=5, color=colors,
               edgecolor='#555', linewidth=0.8, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, color=C_TEXT, fontsize=10)
        ax.set_ylabel(f'Relative L$_2$ error', color=C_TEXT, fontsize=11)
        ax.set_title(f'{ylabel} — Step Neuron Ablation',
                     color=C_TEXT, fontsize=12, fontfamily='serif')
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(j, m + s + 0.005, f'{m:.4f}', ha='center', fontsize=9,
                    color=C_TEXT)
    fig.suptitle('Exp 10B: Riemann Acoustics', color=C_TEXT, fontsize=14,
                 fontfamily='serif', y=1.02)
    plt.tight_layout()
    fname_ab = os.path.join(results_dir, 'exp10b_step_ablation.png')
    plt.savefig(fname_ab, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname_ab}")
    plt.close(fig)

    return {
        'with_steps': {
            'p_l2_mean': float(np.mean(all_p_s)),
            'p_l2_std': float(np.std(all_p_s)),
            'v_l2_mean': float(np.mean(all_v_s)),
            'v_l2_std': float(np.std(all_v_s)),
        },
        'no_steps': {
            'p_l2_mean': float(np.mean(all_p_n)),
            'p_l2_std': float(np.std(all_p_n)),
            'v_l2_mean': float(np.mean(all_v_n)),
            'v_l2_std': float(np.std(all_v_n)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    results_10a = run_exp10a(results_dir)
    results_10b = run_exp10b(results_dir)

    # Save
    all_results = {
        'exp10a_gaussian': results_10a,
        'exp10b_riemann': results_10b,
        'cinn_reference': {
            'note': 'Hidden acoustics (Table 4), v data only, 1000 iter',
            'p_l2': '0.1267+-0.0534', 'v_l2': '0.0588+-0.0077', 'time': '9.1s',
        },
        'pinn_reference': {
            'note': 'Hidden acoustics (Table 4), 5000 iter',
            'p_l2': '0.5209+-0.2774', 'v_l2': '0.0615+-0.0075', 'time': '44.4s',
        },
    }
    out_path = os.path.join(results_dir, 'exp10_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Linear Acoustics")
    print(f"{'='*70}")
    print(f"\n  10A: Gaussian IC (forward problem)")
    r = results_10a
    print(f"    ELM-CINN:  p L2={r['p_l2_mean']:.4f}+-{r['p_l2_std']:.4f}  "
          f"v L2={r['v_l2_mean']:.4f}+-{r['v_l2_std']:.4f}")
    print(f"    CINN:   p L2=0.1267+-0.0534  v L2=0.0588+-0.0077  (hidden, 1000 iter)")
    print(f"    PINN:   p L2=0.5209+-0.2774  v L2=0.0615+-0.0075  (hidden, 5000 iter)")

    print(f"\n  10B: Riemann IC (step neurons)")
    s = results_10b['with_steps']
    n = results_10b['no_steps']
    print(f"    + steps: p L2={s['p_l2_mean']:.4f}+-{s['p_l2_std']:.4f}  "
          f"v L2={s['v_l2_mean']:.4f}+-{s['v_l2_std']:.4f}")
    print(f"    no steps: p L2={n['p_l2_mean']:.4f}+-{n['p_l2_std']:.4f}  "
          f"v L2={n['v_l2_mean']:.4f}+-{n['v_l2_std']:.4f}")


if __name__ == '__main__':
    main()
