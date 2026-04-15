"""
Experiment 9 — ELM-CINN for Periodic Advection
=============================================
Replicates CINN paper (arXiv 2212.14012) Section 4.1.2 and extends
with square wave IC (professor's request).

PDE:   u_t + v * u_x = 0
BC:    u(0,t) = u(2π,t)  (periodic)
Exact: u(x,t) = g((x - v*t) mod 2π)

Sub-experiments:
  9A: IC = sin(x), v = 20,30,40,50  (direct comparison with CINN Table 3)
  9B: IC = square wave, v = 1,5,10,20  (professor's suggestion, step neurons)

ELM-CINN approach:
  - Compute xi = (x - v*t) mod 2pi  (periodic characteristic coordinate)
  - Evaluate basis at xi: same beta from IC fit works for all t
  - For square wave: step neurons capture the 2 discontinuities

Reference (CINN paper Table 3, 10 reps, T=1, 20000 ADAM iterations):
  v=20: CINN L2=0.0300+-0.0043, PINN L2=0.0347+-0.0056
  v=30: CINN L2=0.0579+-0.0095, PINN L2=0.1003+-0.0188
  v=40: CINN L2=0.0852+-0.0169, PINN L2=0.4395+-0.1120
  v=50: CINN L2=0.5365+-0.1729, PINN L2=0.7797+-0.0242
"""

import numpy as np
import matplotlib.pyplot as plt
import json, os, time

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════════════════════════════════════
L = 2 * np.pi    # domain [0, 2π]
T_FINAL = 1.0    # match CINN paper

def sin_ic(x):
    """Smooth IC: u(x,0) = sin(x)."""
    return np.sin(x)

def square_wave_ic(x):
    """Discontinuous IC: u(x,0) = 1 if π/2 < x < 3π/2, else 0."""
    return np.where((x > np.pi/2) & (x < 3*np.pi/2), 1.0, 0.0)

SQUARE_DISC = [np.pi/2, 3*np.pi/2]  # discontinuity positions

def exact_periodic(x, t, v, ic_func):
    """Exact solution: u(x,t) = g((x - v*t) mod 2π)."""
    xi = np.mod(x - v * t, L)
    return ic_func(xi)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Network building blocks
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tanh_weights(n_tanh, seed=7, scale=2.5):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale * L, scale * L, size=n_tanh)
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

def predict(x, W_tanh, b_tanh, positions, kappa, beta):
    H = hidden_matrix(x, W_tanh, b_tanh, positions, kappa)
    return H @ beta

def solve_ridge(H, y, lam=1e-6):
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELM-CINN solver (periodic)
# ═══════════════════════════════════════════════════════════════════════════════
def cielm_periodic(config, ic_func, v, snap_times):
    """
    ELM-CINN for periodic advection.

    Fit IC on [0, 2π], then at any time t evaluate at xi = (x - v*t) mod 2π.
    The mod handles periodicity — no boundary loss needed.
    """
    n_tanh = config['n_tanh']
    kappa = config['kappa']
    seed_tanh = config['seed_tanh']
    positions = np.array(config.get('positions', []))
    K = len(positions)

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed_tanh)

    # Fit IC on [0, 2π] (dense sampling)
    x_ic = np.linspace(0, L, config['n_ic'], endpoint=False)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic)**2)))

    # Evaluate at each snapshot
    x_eval = np.linspace(0, L, config['n_eval'], endpoint=False)
    snapshots = {}

    t_start = time.time()
    for t_snap in snap_times:
        xi = np.mod(x_eval - v * t_snap, L)  # periodic shift
        H_shifted = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_pred = H_shifted @ beta
        u_ref = exact_periodic(x_eval, t_snap, v, ic_func)

        rmse = float(np.sqrt(np.mean((u_pred - u_ref)**2)))
        norm_ref = max(np.linalg.norm(u_ref), 1e-12)
        l1_err = float(np.mean(np.abs(u_pred - u_ref)) /
                        max(np.mean(np.abs(u_ref)), 1e-12))
        l2_err = float(np.linalg.norm(u_pred - u_ref) / norm_ref)

        snapshots[f"t={t_snap:.2f}"] = {
            't': float(t_snap),
            'u_pred': u_pred,
            'u_ref': u_ref,
            'rmse': rmse,
            'l1_error': l1_err,
            'l2_error': l2_err,
        }
    elapsed = time.time() - t_start

    return {
        'ic_rmse': ic_rmse,
        'elapsed_s': elapsed,
        'snapshots': snapshots,
        'x_eval': x_eval,
        'n_params': len(beta),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Visualization (light academic palette)
# ═══════════════════════════════════════════════════════════════════════════════
C_BLUE   = '#2166ac'
C_RED    = '#d6604d'
C_GREEN  = '#1b7837'
C_ORANGE = '#e08214'
C_PURPLE = '#7b3294'
C_CYAN   = '#0571b0'
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

def plot_snapshots(x_eval, snapshots, title, fname):
    snap_keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(snap_keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]

    for i, key in enumerate(snap_keys):
        s = snapshots[key]
        ax = axes[i]
        style_ax(ax)
        ax.plot(x_eval, s['u_ref'], color=C_BLUE, linewidth=2.5, zorder=3,
                label='Exact')
        ax.plot(x_eval, s['u_pred'], color=C_RED, linewidth=2, linestyle='--',
                zorder=4, label='ELM-CINN')
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {s['l2_error']:.4f}",
                     color=C_TEXT, fontsize=11, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')

    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)

def plot_velocity_sweep(results_dict, title, fname, cinn_ref=None, pinn_ref=None):
    """Bar chart: L2 error vs velocity for ELM-CINN, CINN, PINN."""
    velocities = sorted(results_dict.keys())
    cielm_l2 = [results_dict[v]['l2_final'] for v in velocities]
    cielm_std = [results_dict[v].get('l2_std', 0) for v in velocities]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    x_pos = np.arange(len(velocities))
    width = 0.25

    ax.bar(x_pos - width, cielm_l2, width, yerr=cielm_std, capsize=4,
           color=C_GREEN, edgecolor='#555', linewidth=0.8, label='ELM-CINN', zorder=3)

    if cinn_ref:
        cinn_l2 = [cinn_ref[v][0] for v in velocities]
        cinn_std = [cinn_ref[v][1] for v in velocities]
        ax.bar(x_pos, cinn_l2, width, yerr=cinn_std, capsize=4,
               color=C_BLUE, edgecolor='#555', linewidth=0.8, label='CINN', zorder=3)

    if pinn_ref:
        pinn_l2 = [pinn_ref[v][0] for v in velocities]
        pinn_std = [pinn_ref[v][1] for v in velocities]
        ax.bar(x_pos + width, pinn_l2, width, yerr=pinn_std, capsize=4,
               color=C_ORANGE, edgecolor='#555', linewidth=0.8, label='PINN', zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'v={v}' for v in velocities], color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title(title, color=C_TEXT, fontsize=14, fontfamily='serif')
    add_legend(ax)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════
def run_exp9a(results_dir):
    """Exp 9A: sin(x) IC, velocity sweep v=20,30,40,50 (CINN paper comparison)."""
    print(f"\n{'='*70}")
    print(f"  Exp 9A: Periodic Advection, IC=sin(x), velocity sweep")
    print(f"{'='*70}")

    config = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6, 'seed_tanh': 7,
        'positions': [],  # no step neurons for smooth IC
        'n_ic': 500, 'n_eval': 1000,
    }

    velocities = [20, 30, 40, 50]
    snap_times = [0.0, T_FINAL]
    results = {}

    # Run 10 seeds for statistical comparison
    for v in velocities:
        all_l2 = []
        for seed in range(10):
            cfg = {**config, 'seed_tanh': seed}
            res = cielm_periodic(cfg, sin_ic, v, snap_times)
            final_l2 = res['snapshots'][f"t={T_FINAL:.2f}"]['l2_error']
            all_l2.append(final_l2)

        results[v] = {
            'l2_final': float(np.mean(all_l2)),
            'l2_std': float(np.std(all_l2)),
            'l2_all': all_l2,
        }
        print(f"  v={v}: L2={results[v]['l2_final']:.4f}+-{results[v]['l2_std']:.4f}")

    # Show single-seed snapshots for v=30
    res_v30 = cielm_periodic(config, sin_ic, 30,
                              [0.0, 0.25, 0.50, 1.00])
    plot_snapshots(res_v30['x_eval'], res_v30['snapshots'],
                   "Exp 9A: ELM-CINN — Periodic Advection, sin(x), v=30",
                   os.path.join(results_dir, 'exp9a_snapshots_v30.png'))

    return results


def run_exp9b(results_dir):
    """Exp 9B: Square wave IC, velocity sweep (step neurons)."""
    print(f"\n{'='*70}")
    print(f"  Exp 9B: Periodic Advection, IC=square wave, velocity sweep")
    print(f"{'='*70}")

    config = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6, 'seed_tanh': 7,
        'positions': SQUARE_DISC,  # 2 step neurons at discontinuities
        'n_ic': 500, 'n_eval': 1000,
    }
    config_no_steps = {**config, 'positions': []}

    velocities = [1, 5, 10, 20]
    snap_times = [0.0, T_FINAL]
    results_with_steps = {}
    results_no_steps = {}

    for v in velocities:
        all_l2_steps, all_l2_no = [], []
        for seed in range(10):
            cfg_s = {**config, 'seed_tanh': seed}
            cfg_n = {**config_no_steps, 'seed_tanh': seed}
            res_s = cielm_periodic(cfg_s, square_wave_ic, v, snap_times)
            res_n = cielm_periodic(cfg_n, square_wave_ic, v, snap_times)
            all_l2_steps.append(res_s['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])
            all_l2_no.append(res_n['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])

        results_with_steps[v] = {
            'l2_final': float(np.mean(all_l2_steps)),
            'l2_std': float(np.std(all_l2_steps)),
            'l2_all': all_l2_steps,
        }
        results_no_steps[v] = {
            'l2_final': float(np.mean(all_l2_no)),
            'l2_std': float(np.std(all_l2_no)),
            'l2_all': all_l2_no,
        }
        print(f"  v={v}: with steps L2={results_with_steps[v]['l2_final']:.4f}+-"
              f"{results_with_steps[v]['l2_std']:.4f}  |  "
              f"no steps L2={results_no_steps[v]['l2_final']:.4f}+-"
              f"{results_no_steps[v]['l2_std']:.4f}")

    # Snapshots for v=5
    res_sq = cielm_periodic(config, square_wave_ic, 5,
                             [0.0, 0.25, 0.50, 1.00])
    plot_snapshots(res_sq['x_eval'], res_sq['snapshots'],
                   "Exp 9B: ELM-CINN — Periodic Advection, square wave, v=5",
                   os.path.join(results_dir, 'exp9b_snapshots_v5.png'))

    # Snapshots for v=20
    res_sq20 = cielm_periodic(config, square_wave_ic, 20,
                               [0.0, 0.25, 0.50, 1.00])
    plot_snapshots(res_sq20['x_eval'], res_sq20['snapshots'],
                   "Exp 9B: ELM-CINN — Periodic Advection, square wave, v=20",
                   os.path.join(results_dir, 'exp9b_snapshots_v20.png'))

    return results_with_steps, results_no_steps


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Exp 9A: sin IC ──
    results_9a = run_exp9a(results_dir)

    # CINN paper Table 3 reference (mean, std)
    cinn_ref = {20: (0.0300, 0.0043), 30: (0.0579, 0.0095),
                40: (0.0852, 0.0169), 50: (0.5365, 0.1729)}
    pinn_ref = {20: (0.0347, 0.0056), 30: (0.1003, 0.0188),
                40: (0.4395, 0.1120), 50: (0.7797, 0.0242)}

    plot_velocity_sweep(results_9a,
                        "Exp 9A: Periodic Advection sin(x) — L$_2$ vs Velocity",
                        os.path.join(results_dir, 'exp9a_velocity_sweep.png'),
                        cinn_ref, pinn_ref)

    # ── Exp 9B: square wave IC ──
    results_9b_steps, results_9b_no = run_exp9b(results_dir)

    # Comparison chart: with steps vs without steps
    velocities_b = sorted(results_9b_steps.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    x_pos = np.arange(len(velocities_b))
    width = 0.3
    ax.bar(x_pos - width/2,
           [results_9b_steps[v]['l2_final'] for v in velocities_b], width,
           yerr=[results_9b_steps[v]['l2_std'] for v in velocities_b],
           capsize=4, color=C_GREEN, edgecolor='#555', linewidth=0.8,
           label='ELM-CINN + step neurons', zorder=3)
    ax.bar(x_pos + width/2,
           [results_9b_no[v]['l2_final'] for v in velocities_b], width,
           yerr=[results_9b_no[v]['l2_std'] for v in velocities_b],
           capsize=4, color=C_ORANGE, edgecolor='#555', linewidth=0.8,
           label='ELM-CINN (tanh only)', zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'v={v}' for v in velocities_b], color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title('Exp 9B: Square Wave — Step Neurons Ablation',
                 color=C_TEXT, fontsize=14, fontfamily='serif')
    add_legend(ax)
    plt.tight_layout()
    fname_9b = os.path.join(results_dir, 'exp9b_step_ablation.png')
    plt.savefig(fname_9b, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname_9b}")
    plt.close(fig)

    # ── Save results ──
    all_results = {
        'exp9a_sin': {v: {k: vv for k, vv in r.items()}
                      for v, r in results_9a.items()},
        'exp9b_square_with_steps': {v: {k: vv for k, vv in r.items()}
                                     for v, r in results_9b_steps.items()},
        'exp9b_square_no_steps': {v: {k: vv for k, vv in r.items()}
                                   for v, r in results_9b_no.items()},
        'cinn_reference': {str(v): {'l2': m, 'std': s}
                           for v, (m, s) in cinn_ref.items()},
        'pinn_reference': {str(v): {'l2': m, 'std': s}
                           for v, (m, s) in pinn_ref.items()},
    }

    out_path = os.path.join(results_dir, 'exp9_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved: {out_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Periodic Advection")
    print(f"{'='*70}")

    print(f"\n  9A: sin(x) IC")
    print(f"  {'v':<6} {'ELM-CINN':<20} {'CINN':<20} {'PINN':<20}")
    print(f"  {'---':<6} {'----':<20} {'----':<20} {'----':<20}")
    for v in [20, 30, 40, 50]:
        c = results_9a[v]
        print(f"  {v:<6} {c['l2_final']:.4f}+-{c['l2_std']:.4f}       "
              f"{cinn_ref[v][0]:.4f}+-{cinn_ref[v][1]:.4f}       "
              f"{pinn_ref[v][0]:.4f}+-{pinn_ref[v][1]:.4f}")

    print(f"\n  9B: Square wave IC (step neurons ablation)")
    print(f"  {'v':<6} {'ELM-CINN+steps':<20} {'ELM-CINN tanh-only':<20}")
    print(f"  {'---':<6} {'-----------':<20} {'---------------':<20}")
    for v in velocities_b:
        s = results_9b_steps[v]
        n = results_9b_no[v]
        print(f"  {v:<6} {s['l2_final']:.4f}+-{s['l2_std']:.4f}       "
              f"{n['l2_final']:.4f}+-{n['l2_std']:.4f}")


if __name__ == '__main__':
    main()
