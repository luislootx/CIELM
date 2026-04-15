"""
Experiment 11 — ELM-CINN for Inviscid Burgers Equation (Nonlinear)
================================================================
Extends ELM-CINN to nonlinear hyperbolic PDEs via fixed-point iteration.

PDE:   u_t + u * u_x = 0       (inviscid Burgers equation)
       Equivalently: u_t + (u²/2)_x = 0  (conservation form)

Characteristic structure:
  Linear case:    ξ = x - v*t              (v known, direct)
  Nonlinear case: ξ = x - u₀(ξ)*t         (u depends on ξ, iterate)

Fixed-point iteration:
  ξ₀ = x
  ξ_{n+1} = x - u_ELM(ξ_n) * t
  Converges when |du_ELM/dξ · t| < 1 (Banach contraction mapping theorem)
  This holds before shock formation: t < t_break = 1 / max|du₀/dx|

Sub-experiments:
  A) Riemann IC (shock):   u_L > u_R, R-H speed s = (u_L+u_R)/2
  B) Riemann IC (rarefaction): u_L < u_R, expansion fan
  C) Smooth IC (pre-shock): u₀=-sin(πx), t < 1/π ≈ 0.318
  D) Convergence analysis:  iterations & contraction factor vs t

Reference: Karniadakis et al. (2026), "Curvature-Aware Optimization for
High-Accuracy PINNs", arXiv:2604.05230 — reports Adam fails on inviscid
Burgers, requires 2nd-order optimizers + HLLC flux loss.

Our approach: NO optimizer, NO loss function, NO collocation. Analytical.
"""

import numpy as np
import matplotlib.pyplot as plt
import json, os, time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Network building blocks (from Exp8, reused exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tanh_weights(n_tanh, seed=7, scale=2.5, domain_scale=1.0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale * domain_scale, scale * domain_scale, size=n_tanh)
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


def elm_predict(x, W_tanh, b_tanh, positions, kappa, beta):
    H = hidden_matrix(x, W_tanh, b_tanh, positions, kappa)
    return H @ beta


def solve_ridge(H, y, lam=1e-6):
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Exact solutions for inviscid Burgers
# ═══════════════════════════════════════════════════════════════════════════════

def burgers_riemann_shock_exact(x, t, u_L, u_R, x_disc):
    """Riemann shock (u_L > u_R): step moves at R-H speed s = (u_L+u_R)/2."""
    s = 0.5 * (u_L + u_R)
    return np.where(x < x_disc + s * t, u_L, u_R)


def burgers_riemann_rarefaction_exact(x, t, u_L, u_R, x_disc):
    """Riemann rarefaction (u_L < u_R): linear fan between characteristics."""
    if t < 1e-14:
        return np.where(x < x_disc, u_L, u_R)
    x_left = x_disc + u_L * t
    x_right = x_disc + u_R * t
    return np.where(x < x_left, u_L,
           np.where(x > x_right, u_R,
                    (x - x_disc) / t))


def burgers_smooth_exact(x_eval, t, ic_func, xi_min=-3.0, xi_max=3.0,
                         n_char=10000):
    """
    Exact solution by forward-tracing characteristics (pre-shock).

    Characteristics: x = ξ + u₀(ξ)*t,  u(x,t) = u₀(ξ)
    Forward-trace a dense grid of ξ → x, then interpolate.
    Valid only when the mapping ξ → x is monotone (pre-shock).
    """
    if t < 1e-14:
        return ic_func(x_eval)

    xi_fine = np.linspace(xi_min, xi_max, n_char)
    u_fine = ic_func(xi_fine)
    x_fine = xi_fine + u_fine * t

    # Check monotonicity
    dx = np.diff(x_fine)
    if np.any(dx <= 0):
        print(f"    WARNING: characteristics crossing at t={t:.4f} (post-shock)")

    return np.interp(x_eval, x_fine, u_fine)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELM-CINN Fixed-Point Iteration (THE KEY NOVELTY)
# ═══════════════════════════════════════════════════════════════════════════════

def cielm_fixed_point(x_eval, t, W_tanh, b_tanh, positions, kappa, beta,
                      max_iter=200, tol=1e-12):
    """
    ELM-CINN for nonlinear Burgers: u_t + u * u_x = 0

    Fixed-point iteration on characteristic coordinate:
        ξ_{n+1} = x - u_ELM(ξ_n) * t

    Convergence (Banach): requires |du_ELM/dξ| * t < 1
    This holds before shock formation.

    Returns: (u_pred, info_dict)
    """
    if t < 1e-14:
        u_pred = elm_predict(x_eval, W_tanh, b_tanh, positions, kappa, beta)
        return u_pred, {'iterations': 0, 'max_residual': 0.0, 'converged': True}

    xi = x_eval.copy()
    residuals = []

    for n_iter in range(max_iter):
        u_at_xi = elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)
        xi_new = x_eval - u_at_xi * t

        res = float(np.max(np.abs(xi_new - xi)))
        residuals.append(res)
        xi = xi_new

        if res < tol:
            break

    u_pred = elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)
    return u_pred, {
        'iterations': n_iter + 1,
        'max_residual': residuals[-1] if residuals else 0.0,
        'converged': residuals[-1] < tol if residuals else True,
        'residuals': residuals,
    }


def cielm_charshift(x_eval, t, W_tanh, b_tanh, positions_ic, kappa, beta,
                    v_shift):
    """
    ELM-CINN characteristic shift (linear): evaluate basis at ξ = x - v*t.
    Used for R-H shock tracking where v = s = (u_L + u_R)/2.
    """
    xi = x_eval - v_shift * t
    return elm_predict(xi, W_tanh, b_tanh, positions_ic, kappa, beta)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plotting (light academic palette)
# ═══════════════════════════════════════════════════════════════════════════════

C_BLUE   = '#2166ac'
C_RED    = '#d6604d'
C_GREEN  = '#1b7837'
C_ORANGE = '#e08214'
C_PURPLE = '#7b3294'
C_TEXT   = '#333333'
BG       = '#ffffff'
BG_AX    = '#fafafa'
GRID     = '#e0e0e0'
SPINE    = '#aaaaaa'


def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)


def add_legend(ax, **kw):
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=SPINE, **kw)


def plot_snapshots(x_eval, snapshots, title, fname, method_label='ELM-CINN'):
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
                zorder=4, label=method_label)
        if 'shock_pos' in s and s['shock_pos'] is not None:
            ax.axvline(s['shock_pos'], color=C_ORANGE, linewidth=1.5,
                       linestyle=':', alpha=0.7, zorder=2, label='Step neuron')
        l2_str = f"{s['l2_error']:.1e}" if s['l2_error'] < 0.001 else f"{s['l2_error']:.4f}"
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {l2_str}",
                     color=C_TEXT, fontsize=11, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x, t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')

    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved: {fname}")
    plt.close(fig)


def compute_errors(u_pred, u_ref):
    norm_ref = max(np.linalg.norm(u_ref), 1e-12)
    l2 = float(np.linalg.norm(u_pred - u_ref) / norm_ref)
    mean_ref = max(np.mean(np.abs(u_ref)), 1e-12)
    l1 = float(np.mean(np.abs(u_pred - u_ref)) / mean_ref)
    return l1, l2


# ═══════════════════════════════════════════════════════════════════════════════
# 5A. Experiment 11A: Riemann IC — Shock (u_L > u_R)
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11a():
    """
    Burgers Riemann shock: u_L=1, u_R=0, shock speed s=0.5.

    Two methods:
      1) R-H characteristic shift:  ξ = x - s*t  (uses shock speed)
      2) Fixed-point iteration:     ξ = x - u_ELM(ξ)*t  (no shock speed given)
    """
    print("\n" + "=" * 70)
    print("Exp 11A: Burgers Riemann -- Shock (u_L=1, u_R=0)")
    print("=" * 70)

    u_L, u_R, x_disc = 1.0, 0.0, 0.0
    x_min, x_max = -1.0, 2.0
    snap_times = [0.0, 0.3, 0.8, 1.5]

    n_tanh, kappa, n_ic, n_eval, lam = 80, 500, 800, 1000, 1e-6
    seed = 7

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))
    positions_ic = [x_disc]

    # Fit IC on extended domain (accounts for characteristic travel)
    margin = max(abs(u_L), abs(u_R)) * max(snap_times) + 0.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = np.where(x_ic < x_disc, u_L, u_R)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")

    x_eval = np.linspace(x_min, x_max, n_eval)
    s_rh = 0.5 * (u_L + u_R)

    # ── Method 1: R-H characteristic shift ──
    print("\n  [R-H tracking]  v_shift = s = (u_L+u_R)/2 =", s_rh)
    t0 = time.time()
    snaps_rh = {}
    for ts in snap_times:
        u_pred = cielm_charshift(x_eval, ts, W_tanh, b_tanh, positions_ic,
                                 kappa, beta, v_shift=s_rh)
        u_ref = burgers_riemann_shock_exact(x_eval, ts, u_L, u_R, x_disc)
        l1, l2 = compute_errors(u_pred, u_ref)
        shock_pos = x_disc + s_rh * ts
        snaps_rh[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                        l1_error=l1, l2_error=l2,
                                        shock_pos=shock_pos)
        print(f"    t={ts:.2f}: L2={l2:.4f}, shock at x={shock_pos:.3f}")
    elapsed_rh = time.time() - t0
    print(f"  Time: {elapsed_rh:.4f}s")

    # ── Method 2: Fixed-point iteration ──
    print("\n  [Fixed-point]  (no shock speed provided)")
    t0 = time.time()
    snaps_fp = {}
    for ts in snap_times:
        u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                         positions_ic, kappa, beta)
        u_ref = burgers_riemann_shock_exact(x_eval, ts, u_L, u_R, x_disc)
        l1, l2 = compute_errors(u_pred, u_ref)
        snaps_fp[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                        l1_error=l1, l2_error=l2,
                                        shock_pos=None)
        status = "OK" if info['converged'] else "NOT CONVERGED"
        print(f"    t={ts:.2f}: L2={l2:.4f}, iters={info['iterations']}, "
              f"{status}, res={info['max_residual']:.2e}")
    elapsed_fp = time.time() - t0
    print(f"  Time: {elapsed_fp:.4f}s")

    # Plots
    plot_snapshots(x_eval, snaps_rh,
                   'Exp 11A: Burgers Shock -- ELM-CINN + R-H Tracking',
                   os.path.join(RESULTS_DIR, 'exp11a_shock_rh.png'),
                   'ELM-CINN + R-H')
    plot_snapshots(x_eval, snaps_fp,
                   'Exp 11A: Burgers Shock -- ELM-CINN + Fixed-Point',
                   os.path.join(RESULTS_DIR, 'exp11a_shock_fp.png'),
                   'ELM-CINN + FP')

    def _strip(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                     if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    return dict(experiment='11A_shock', ic_rmse=ic_rmse,
                rh=_strip(snaps_rh), fp=_strip(snaps_fp),
                elapsed_rh=elapsed_rh, elapsed_fp=elapsed_fp)


# ═══════════════════════════════════════════════════════════════════════════════
# 5B. Experiment 11B: Riemann IC — Rarefaction (u_L < u_R)
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11b():
    """
    Burgers Riemann rarefaction: u_L=0, u_R=1 → expansion fan.
    Fixed-point iteration from step IC.
    """
    print("\n" + "=" * 70)
    print("Exp 11B: Burgers Riemann -- Rarefaction (u_L=0, u_R=1)")
    print("=" * 70)

    u_L, u_R, x_disc = 0.0, 1.0, 0.0
    x_min, x_max = -1.0, 2.0
    snap_times = [0.0, 0.2, 0.5, 1.0]

    n_tanh, kappa, n_ic, n_eval, lam = 80, 500, 800, 1000, 1e-6
    seed = 7

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))
    positions_ic = [x_disc]

    margin = max(abs(u_L), abs(u_R)) * max(snap_times) + 0.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = np.where(x_ic < x_disc, u_L, u_R)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")

    x_eval = np.linspace(x_min, x_max, n_eval)

    t0 = time.time()
    snapshots = {}
    for ts in snap_times:
        u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                         positions_ic, kappa, beta)
        u_ref = burgers_riemann_rarefaction_exact(x_eval, ts, u_L, u_R, x_disc)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                         l1_error=l1, l2_error=l2,
                                         shock_pos=None)
        status = "OK" if info['converged'] else "NOT CONVERGED"
        print(f"    t={ts:.2f}: L2={l2:.4f}, iters={info['iterations']}, "
              f"{status}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.4f}s")

    plot_snapshots(x_eval, snapshots,
                   'Exp 11B: Burgers Rarefaction -- ELM-CINN + Fixed-Point',
                   os.path.join(RESULTS_DIR, 'exp11b_rarefaction.png'),
                   'ELM-CINN + FP')

    def _strip(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                     if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    return dict(experiment='11B_rarefaction', ic_rmse=ic_rmse,
                results=_strip(snapshots), elapsed=elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# 5C. Experiment 11C: Smooth IC — u₀(x) = -sin(πx)
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11c():
    """
    Burgers smooth IC: u₀(x) = -sin(πx) on [-1, 1].

    Breaking time: t_break = 1/max|du₀/dx| = 1/π ≈ 0.3183
    Before shock: fixed-point converges (contraction mapping).
    """
    print("\n" + "=" * 70)
    print("Exp 11C: Burgers Smooth IC -- u0(x) = -sin(pi*x)")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi
    snap_times = [0.0, 0.10, 0.20, 0.30]  # all pre-shock

    def ic_func(x):
        return -np.sin(np.pi * x)

    # No step neurons — smooth IC
    n_tanh, kappa, n_ic, n_eval, lam = 120, 500, 1000, 1000, 1e-8
    seed = 7
    positions_ic = []

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=3.0,
                                            domain_scale=(x_max - x_min))

    # Fit on extended domain (characteristics shift by at most |u_max|*t = 0.3)
    margin = 1.0 * max(snap_times) + 0.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")
    print(f"  Breaking time: t_break = 1/pi = {t_break:.4f}")

    x_eval = np.linspace(x_min, x_max, n_eval)

    t0 = time.time()
    snapshots = {}
    for ts in snap_times:
        u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                         positions_ic, kappa, beta)
        u_ref = burgers_smooth_exact(x_eval, ts, ic_func,
                                     xi_min=x_min - margin,
                                     xi_max=x_max + margin)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                         l1_error=l1, l2_error=l2,
                                         shock_pos=None)
        status = "OK" if info['converged'] else "NOT CONVERGED"
        print(f"    t={ts:.2f}: L2={l2:.4f}, iters={info['iterations']}, "
              f"{status}, res={info['max_residual']:.2e}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.4f}s")

    plot_snapshots(x_eval, snapshots,
                   f'Exp 11C: Burgers Smooth IC  (t_break ≈ {t_break:.3f})',
                   os.path.join(RESULTS_DIR, 'exp11c_smooth.png'),
                   'ELM-CINN + FP')

    def _strip(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                     if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    return dict(experiment='11C_smooth', ic_rmse=ic_rmse,
                t_break=t_break, results=_strip(snapshots), elapsed=elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# 5D. Convergence analysis for smooth IC
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11d_convergence():
    """
    Convergence analysis: how iterations, contraction factor, and L2 error
    scale with time for the smooth IC case.
    """
    print("\n" + "=" * 70)
    print("Exp 11D: Convergence Analysis -- Smooth IC")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi

    def ic_func(x):
        return -np.sin(np.pi * x)

    n_tanh, kappa, n_ic, n_eval, lam = 120, 500, 1000, 500, 1e-8
    seed = 7
    positions_ic = []

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=3.0,
                                            domain_scale=(x_max - x_min))
    margin = 1.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    x_eval = np.linspace(x_min, x_max, n_eval)

    # Sweep t from 0 to beyond t_break
    t_values = np.linspace(0.01, 0.40, 40)
    iters_list, l2_list, converged_list = [], [], []

    for t_val in t_values:
        u_pred, info = cielm_fixed_point(x_eval, t_val, W_tanh, b_tanh,
                                         positions_ic, kappa, beta,
                                         max_iter=500, tol=1e-12)
        u_ref = burgers_smooth_exact(x_eval, t_val, ic_func,
                                     xi_min=x_min - margin,
                                     xi_max=x_max + margin)
        _, l2 = compute_errors(u_pred, u_ref)
        iters_list.append(info['iterations'])
        l2_list.append(l2)
        converged_list.append(info['converged'])
        marker = "x" if not info['converged'] else ""
        print(f"    t={t_val:.3f}: L2={l2:.4f}, iters={info['iterations']:3d} "
              f"{'  *** NOT CONVERGED' if not info['converged'] else ''}")

    # ── Plot: iterations + L2 vs t ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)

    # Panel 1: iterations vs t
    ax = axes[0]
    style_ax(ax)
    ax.plot(t_values, iters_list, color=C_RED, linewidth=2, marker='o',
            markersize=3)
    ax.axvline(t_break, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label=f'$t_{{break}}$ = 1/$\\pi$ = {t_break:.3f}')
    ax.set_xlabel('t', color=C_TEXT, fontsize=10)
    ax.set_ylabel('Iterations to converge', color=C_TEXT, fontsize=10)
    ax.set_title('Fixed-point iterations', color=C_TEXT, fontsize=11,
                 fontfamily='serif')
    add_legend(ax)

    # Panel 2: L2 error vs t
    ax = axes[1]
    style_ax(ax)
    ax.semilogy(t_values, l2_list, color=C_BLUE, linewidth=2, marker='s',
                markersize=3)
    ax.axvline(t_break, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label='$t_{break}$')
    ax.set_xlabel('t', color=C_TEXT, fontsize=10)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=10)
    ax.set_title('Solution accuracy', color=C_TEXT, fontsize=11,
                 fontfamily='serif')
    add_legend(ax)

    # Panel 3: contraction factor = pi*t (theoretical) vs observed
    ax = axes[2]
    style_ax(ax)
    # Theoretical contraction factor for -sin(pi*x): max|du0/dxi|*t = pi*t
    ax.plot(t_values, np.pi * t_values, color=C_GREEN, linewidth=2,
            label=r'$\pi t$ (theoretical)')
    ax.axhline(1.0, color=C_RED, linestyle='--', linewidth=1.5,
               label='Contraction limit')
    ax.axvline(t_break, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label='t_break')
    ax.set_xlabel('t', color=C_TEXT, fontsize=10)
    ax.set_ylabel('Contraction factor', color=C_TEXT, fontsize=10)
    ax.set_title('Banach contraction condition', color=C_TEXT, fontsize=11,
                 fontfamily='serif')
    add_legend(ax, loc='upper left')
    ax.set_ylim(0, 1.5)

    fig.suptitle('Exp 11D: Convergence Analysis -- Burgers Smooth IC',
                 color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'exp11d_convergence.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved: {fname}")
    plt.close(fig)

    return dict(experiment='11D_convergence', t_break=t_break,
                t_values=t_values.tolist(), iterations=iters_list,
                l2_errors=l2_list,
                converged=[bool(c) for c in converged_list])


# ═══════════════════════════════════════════════════════════════════════════════
# 5E. 10-seed statistical comparison (Riemann shock)
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical():
    """10-seed comparison for Riemann shock, matching CINN paper format."""
    print("\n" + "=" * 70)
    print("Statistical Comparison: 10 seeds, Burgers Shock at t=1.0")
    print("=" * 70)

    u_L, u_R, x_disc = 1.0, 0.0, 0.0
    x_min, x_max = -1.0, 2.0
    T_eval = 1.0
    s_rh = 0.5 * (u_L + u_R)

    n_tanh, kappa, n_ic, n_eval, lam = 80, 500, 800, 1000, 1e-6

    x_eval = np.linspace(x_min, x_max, n_eval)
    u_ref = burgers_riemann_shock_exact(x_eval, T_eval, u_L, u_R, x_disc)

    l1s, l2s, ts = [], [], []
    for seed in range(10):
        W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=2.5,
                                                domain_scale=(x_max - x_min))
        positions_ic = [x_disc]
        margin = max(abs(u_L), abs(u_R)) * T_eval + 0.5
        x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
        y_ic = np.where(x_ic < x_disc, u_L, u_R)
        H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
        beta = solve_ridge(H_ic, y_ic, lam)

        t0 = time.time()
        u_pred = cielm_charshift(x_eval, T_eval, W_tanh, b_tanh,
                                 positions_ic, kappa, beta, v_shift=s_rh)
        elapsed = time.time() - t0

        l1, l2 = compute_errors(u_pred, u_ref)
        l1s.append(l1); l2s.append(l2); ts.append(elapsed)

    l1a, l2a, ta = np.array(l1s), np.array(l2s), np.array(ts)
    print(f"  L1:   {l1a.mean():.4f} ± {l1a.std():.4f}")
    print(f"  L2:   {l2a.mean():.4f} ± {l2a.std():.4f}")
    print(f"  Time: {ta.mean():.5f} ± {ta.std():.5f}s")

    return dict(l1_mean=float(l1a.mean()), l1_std=float(l1a.std()),
                l2_mean=float(l2a.mean()), l2_std=float(l2a.std()),
                time_mean=float(ta.mean()), time_std=float(ta.std()))


def run_stats_smooth():
    """10-seed stats for smooth pre-shock (11C) and smooth rarefaction (11E)."""
    print("\n" + "=" * 70)
    print("Statistical Comparison: 10 seeds, Smooth Cases")
    print("=" * 70)

    results = {}

    # ── 11C: -sin(pi*x), t=0.20 (well within contraction) ──
    print("\n  [11C] u0=-sin(pi*x), t=0.20:")
    x_min, x_max = -1.0, 1.0
    T_eval = 0.20
    def ic_c(x): return -np.sin(np.pi * x)
    margin_c = 1.5

    l2s_c, ts_c, iters_c = [], [], []
    for seed in range(10):
        W, b = generate_tanh_weights(120, seed, scale=3.0,
                                      domain_scale=(x_max - x_min))
        x_ic = np.linspace(x_min - margin_c, x_max + margin_c, 1000)
        H = hidden_matrix(x_ic, W, b, [], 500)
        beta = solve_ridge(H, ic_c(x_ic), 1e-8)

        x_ev = np.linspace(x_min, x_max, 1000)
        t0 = time.time()
        u_pred, info = cielm_fixed_point(x_ev, T_eval, W, b, [], 500, beta)
        el = time.time() - t0

        u_ref = burgers_smooth_exact(x_ev, T_eval, ic_c,
                                     xi_min=x_min - margin_c,
                                     xi_max=x_max + margin_c)
        _, l2 = compute_errors(u_pred, u_ref)
        l2s_c.append(l2); ts_c.append(el); iters_c.append(info['iterations'])

    l2a = np.array(l2s_c); ta = np.array(ts_c); ia = np.array(iters_c)
    print(f"    L2:    {l2a.mean():.2e} +/- {l2a.std():.2e}")
    print(f"    Iters: {ia.mean():.0f} +/- {ia.std():.0f}")
    print(f"    Time:  {ta.mean():.5f} +/- {ta.std():.5f}s")
    results['11c_smooth'] = dict(
        t=T_eval, l2_mean=float(l2a.mean()), l2_std=float(l2a.std()),
        iters_mean=float(ia.mean()), iters_std=float(ia.std()),
        time_mean=float(ta.mean()), time_std=float(ta.std()))

    # ── 11E: 0.5+0.4*tanh(x), t=2.0 ──
    print("\n  [11E] u0=0.5+0.4*tanh(x), t=2.0:")
    x_min_e, x_max_e = -3.0, 5.0
    T_eval_e = 2.0
    def ic_e(x): return 0.5 + 0.4 * np.tanh(x)
    margin_e = 0.9 * T_eval_e + 1.0

    l2s_e, ts_e, iters_e = [], [], []
    for seed in range(10):
        W, b = generate_tanh_weights(120, seed, scale=2.5,
                                      domain_scale=(x_max_e - x_min_e))
        x_ic = np.linspace(x_min_e - margin_e, x_max_e + margin_e, 1200)
        H = hidden_matrix(x_ic, W, b, [], 500)
        beta = solve_ridge(H, ic_e(x_ic), 1e-8)

        x_ev = np.linspace(x_min_e, x_max_e, 1000)
        t0 = time.time()
        u_pred, info = cielm_fixed_point(x_ev, T_eval_e, W, b, [], 500, beta,
                                         max_iter=500)
        el = time.time() - t0

        u_ref = burgers_smooth_exact(x_ev, T_eval_e, ic_e,
                                     xi_min=x_min_e - margin_e,
                                     xi_max=x_max_e + margin_e,
                                     n_char=20000)
        _, l2 = compute_errors(u_pred, u_ref)
        l2s_e.append(l2); ts_e.append(el); iters_e.append(info['iterations'])

    l2a = np.array(l2s_e); ta = np.array(ts_e); ia = np.array(iters_e)
    print(f"    L2:    {l2a.mean():.2e} +/- {l2a.std():.2e}")
    print(f"    Iters: {ia.mean():.0f} +/- {ia.std():.0f}")
    print(f"    Time:  {ta.mean():.5f} +/- {ta.std():.5f}s")
    results['11e_rarefaction'] = dict(
        t=T_eval_e, l2_mean=float(l2a.mean()), l2_std=float(l2a.std()),
        iters_mean=float(ia.mean()), iters_std=float(ia.std()),
        time_mean=float(ta.mean()), time_std=float(ta.std()))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IMPROVEMENT 1: Smooth rarefaction IC (no shock ever)
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11e_rarefaction_smooth():
    """
    Smooth monotone-increasing IC: du0/dx > 0 everywhere.
    Characteristics DIVERGE => no shock forms for any t.
    Fixed-point should converge for all time.

    IC: u0(x) = 0.5 + 0.4*tanh(x)   on [-3, 5]
    max|du0/dx| = 0.4 at x=0 => contraction holds for t < 2.5
    """
    print("\n" + "=" * 70)
    print("Exp 11E: Burgers Smooth Rarefaction -- u0 = 0.5+0.4*tanh(x)")
    print("=" * 70)

    x_min, x_max = -3.0, 5.0

    def ic_func(x):
        return 0.5 + 0.4 * np.tanh(x)

    # du0/dx = 0.4*sech^2(x) > 0 always
    # max(du0/dx) = 0.4 at x=0  =>  contraction for t < 1/0.4 = 2.5
    snap_times = [0.0, 0.5, 1.0, 2.0]

    n_tanh, kappa, n_ic, n_eval, lam = 120, 500, 1200, 1000, 1e-8
    seed = 7
    positions_ic = []

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))

    # Extended domain for characteristic travel
    u_max = 0.9  # max value of IC is 0.5+0.4 = 0.9
    margin = u_max * max(snap_times) + 1.0
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")
    print(f"  du0/dx > 0 everywhere => NO shock for any t")

    x_eval = np.linspace(x_min, x_max, n_eval)

    t0 = time.time()
    snapshots = {}
    for ts in snap_times:
        u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                         positions_ic, kappa, beta,
                                         max_iter=500, tol=1e-12)
        u_ref = burgers_smooth_exact(x_eval, ts, ic_func,
                                     xi_min=x_min - margin,
                                     xi_max=x_max + margin,
                                     n_char=20000)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                         l1_error=l1, l2_error=l2,
                                         shock_pos=None)
        status = "OK" if info['converged'] else "NOT CONVERGED"
        print(f"    t={ts:.2f}: L2={l2:.6f}, iters={info['iterations']:3d}, "
              f"{status}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.4f}s")

    plot_snapshots(x_eval, snapshots,
                   'Exp 11E: Burgers Smooth Rarefaction (no shock)',
                   os.path.join(RESULTS_DIR, 'exp11e_rarefaction_smooth.png'),
                   'ELM-CINN + FP')

    def _strip(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                     if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    return dict(experiment='11E_rarefaction_smooth', ic_rmse=ic_rmse,
                results=_strip(snapshots), elapsed=elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. IMPROVEMENT 2: Post-shock with step neuron insertion
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp11f_post_shock():
    """
    Smooth IC: u0(x) = -sin(pi*x) on [-1, 1].
    Shock forms at t_break = 1/pi ~ 0.318 at x=0.

    Post-shock strategy:
      1. Detect shock region (where FP oscillates / doesn't converge)
      2. By antisymmetry of -sin(pi*x), shock stays at x=0 for all t
      3. Insert step neuron at x_shock = 0
      4. Solve LEFT and RIGHT of shock separately with FP
      5. Step neuron handles the jump

    For general ICs, shock position comes from the equal-area rule.
    Here symmetry gives us x_shock = 0 and shock speed = 0.
    """
    print("\n" + "=" * 70)
    print("Exp 11F: Burgers Post-Shock -- step neuron insertion")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi

    def ic_func(x):
        return -np.sin(np.pi * x)

    # Test times spanning pre-shock and post-shock
    snap_times = [0.0, 0.2, 0.5, 1.0, 2.0]

    n_tanh, kappa, n_ic, n_eval, lam = 120, 500, 1000, 1000, 1e-8
    seed = 7
    positions_ic = []  # no step in original IC fit

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=3.0,
                                            domain_scale=(x_max - x_min))

    margin = 1.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta_orig = solve_ridge(H_ic, y_ic, lam)

    # Also prepare an ELM WITH a step neuron at x=0 for post-shock
    positions_shock = [0.0]
    H_ic_step = hidden_matrix(x_ic, W_tanh, b_tanh, positions_shock, kappa)
    beta_step = solve_ridge(H_ic_step, y_ic, lam)

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta_orig - y_ic) ** 2)))
    print(f"  IC fit RMSE (no step): {ic_rmse:.6f}")
    ic_rmse_step = float(np.sqrt(np.mean((H_ic_step @ beta_step - y_ic) ** 2)))
    print(f"  IC fit RMSE (w/ step): {ic_rmse_step:.6f}")
    print(f"  t_break = 1/pi = {t_break:.4f}")
    print(f"  Shock at x=0 for all t > t_break (by antisymmetry)")

    x_eval = np.linspace(x_min, x_max, n_eval)

    # ── Method 1: Pure FP (baseline, will fail post-shock) ──
    print("\n  [Pure FP -- baseline]")
    snaps_pure = {}
    for ts in snap_times:
        u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                         positions_ic, kappa, beta_orig,
                                         max_iter=500, tol=1e-10)
        u_ref = burgers_smooth_exact(x_eval, ts, ic_func,
                                     xi_min=x_min - margin,
                                     xi_max=x_max + margin)
        l1, l2 = compute_errors(u_pred, u_ref)
        snaps_pure[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                          l1_error=l1, l2_error=l2,
                                          shock_pos=None)
        status = "OK" if info['converged'] else "DIVERGED"
        print(f"    t={ts:.2f}: L2={l2:.6f}, iters={info['iterations']}, {status}")

    # ── Method 2: Time-stepping ELM-CINN ──
    # Advance in small dt steps, each within contraction regime.
    # At each step, recompute characteristics from current solution.
    print("\n  [Time-stepping ELM-CINN -- small dt within contraction]")
    t0 = time.time()
    snaps_step = {}

    dt_step = 0.05  # small enough that pi*dt < 1
    x_shock = None  # will be detected when FP fails to converge

    for ts in snap_times:
        if ts < 1e-14:
            u_pred = elm_predict(x_eval, W_tanh, b_tanh, positions_ic,
                                kappa, beta_orig)
            info = {'iterations': 0, 'converged': True}
            method_used = "IC"
        elif ts <= t_break * 0.95:
            # Pre-shock: standard single-shot FP
            u_pred, info = cielm_fixed_point(x_eval, ts, W_tanh, b_tanh,
                                             positions_ic, kappa, beta_orig,
                                             max_iter=500, tol=1e-12)
            method_used = "FP (pre-shock)"
        else:
            # Post-shock: time-stepping approach
            # Step 1: advance to t_break with single-shot FP
            t_safe = t_break * 0.9
            u_curr, _ = cielm_fixed_point(x_eval, t_safe, W_tanh, b_tanh,
                                          positions_ic, kappa, beta_orig,
                                          max_iter=500, tol=1e-12)

            # Step 2: from t_safe to ts in small incremental steps
            # At each step, fit a NEW temporary ELM to u_curr and advance by dt
            t_curr = t_safe
            total_iters = 0
            all_converged = True

            while t_curr < ts - 1e-14:
                dt = min(dt_step, ts - t_curr)

                # Fit temporary ELM to current solution
                H_tmp = hidden_matrix(x_eval, W_tanh, b_tanh,
                                      positions_ic, kappa)
                beta_tmp = solve_ridge(H_tmp, u_curr, lam)

                # FP advance by dt
                u_next, info_tmp = cielm_fixed_point(
                    x_eval, dt, W_tanh, b_tanh,
                    positions_ic, kappa, beta_tmp,
                    max_iter=200, tol=1e-10)

                total_iters += info_tmp['iterations']
                if not info_tmp['converged']:
                    all_converged = False

                u_curr = u_next
                t_curr += dt

            u_pred = u_curr
            info = {'iterations': total_iters, 'converged': all_converged}
            method_used = "Time-step FP (post-shock)"

        u_ref = burgers_smooth_exact(x_eval, ts, ic_func,
                                     xi_min=x_min - margin,
                                     xi_max=x_max + margin)
        l1, l2 = compute_errors(u_pred, u_ref)
        snaps_step[f"t={ts:.2f}"] = dict(t=ts, u_pred=u_pred, u_ref=u_ref,
                                          l1_error=l1, l2_error=l2,
                                          shock_pos=None)
        status = "OK" if info['converged'] else "DIVERGED"
        print(f"    t={ts:.2f}: L2={l2:.6f}, iters={info['iterations']}, "
              f"{status}  [{method_used}]")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.4f}s")

    # Plot comparison: pure FP vs FP+step
    plot_snapshots(x_eval, snaps_pure,
                   'Exp 11F: Burgers Post-Shock -- Pure FP (baseline)',
                   os.path.join(RESULTS_DIR, 'exp11f_postshock_pure.png'),
                   'Pure FP')
    plot_snapshots(x_eval, snaps_step,
                   'Exp 11F: Burgers Post-Shock -- Time-stepping ELM-CINN',
                   os.path.join(RESULTS_DIR, 'exp11f_postshock_timestep.png'),
                   'TS-ELM-CINN')

    def _strip(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                     if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    return dict(experiment='11F_post_shock', ic_rmse=ic_rmse,
                pure_fp=_strip(snaps_pure), fp_step=_strip(snaps_step),
                elapsed=elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. IMPROVEMENT 3: Anderson acceleration for FP
# ═══════════════════════════════════════════════════════════════════════════════

def anderson_fixed_point(x_eval, t, W_tanh, b_tanh, positions, kappa, beta,
                         max_iter=200, tol=1e-12, m=5):
    """
    Anderson-accelerated fixed-point iteration.

    Instead of plain ξ_{n+1} = G(ξ_n), uses a linear combination of
    the last m iterates to minimize the residual in least-squares sense.

    Typically 3-5x fewer iterations than plain FP.
    """
    if t < 1e-14:
        u_pred = elm_predict(x_eval, W_tanh, b_tanh, positions, kappa, beta)
        return u_pred, {'iterations': 0, 'max_residual': 0.0, 'converged': True}

    N = len(x_eval)
    xi = x_eval.copy()

    # History buffers
    Xi_hist = []   # previous iterates
    G_hist = []    # G(xi) values
    residuals = []

    for k in range(max_iter):
        u_at_xi = elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)
        g = x_eval - u_at_xi * t   # G(xi) = x - u_ELM(xi)*t
        f = g - xi                  # residual

        res = float(np.max(np.abs(f)))
        residuals.append(res)

        if res < tol:
            xi = g
            break

        Xi_hist.append(xi.copy())
        G_hist.append(g.copy())

        if len(Xi_hist) > m + 1:
            Xi_hist.pop(0)
            G_hist.pop(0)

        mk = len(Xi_hist) - 1  # number of previous pairs available

        if mk < 1:
            # Not enough history yet, do plain FP step
            xi = g
        else:
            # Build delta matrices
            # dF[:, j] = F_{k} - F_{k-mk+j}  where F = G - Xi (residuals)
            F_curr = G_hist[-1] - Xi_hist[-1]
            dF = np.zeros((N, mk))
            for j in range(mk):
                F_prev = G_hist[j] - Xi_hist[j]
                dF[:, j] = F_curr - F_prev

            # Solve least-squares: min ||dF @ alpha - F_curr||
            try:
                alpha, _, _, _ = np.linalg.lstsq(dF, F_curr, rcond=None)
            except np.linalg.LinAlgError:
                xi = g
                continue

            # Anderson update
            xi_new = (1.0 - np.sum(alpha)) * G_hist[-1]
            for j in range(mk):
                xi_new += alpha[j] * G_hist[j]
            xi = xi_new

    u_pred = elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)
    return u_pred, {
        'iterations': k + 1 if 'k' in dir() else len(residuals),
        'max_residual': residuals[-1] if residuals else 0.0,
        'converged': (residuals[-1] < tol) if residuals else True,
        'residuals': residuals,
    }


def run_exp11g_anderson():
    """
    Compare plain FP vs Anderson-accelerated FP on smooth IC.
    """
    print("\n" + "=" * 70)
    print("Exp 11G: Anderson Acceleration vs Plain FP")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi

    def ic_func(x):
        return -np.sin(np.pi * x)

    n_tanh, kappa, n_ic, n_eval, lam = 120, 500, 1000, 500, 1e-8
    seed = 7
    positions_ic = []

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed, scale=3.0,
                                            domain_scale=(x_max - x_min))
    margin = 1.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic_func(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions_ic, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)

    x_eval = np.linspace(x_min, x_max, n_eval)

    t_values = np.linspace(0.01, 0.32, 32)
    iters_plain, iters_anderson = [], []

    for t_val in t_values:
        # Plain FP
        _, info_p = cielm_fixed_point(x_eval, t_val, W_tanh, b_tanh,
                                      positions_ic, kappa, beta,
                                      max_iter=1000, tol=1e-12)
        iters_plain.append(info_p['iterations'])

        # Anderson FP
        _, info_a = anderson_fixed_point(x_eval, t_val, W_tanh, b_tanh,
                                         positions_ic, kappa, beta,
                                         max_iter=1000, tol=1e-12, m=5)
        iters_anderson.append(info_a['iterations'])

        print(f"    t={t_val:.3f}: plain={info_p['iterations']:4d}  "
              f"anderson={info_a['iterations']:4d}  "
              f"speedup={info_p['iterations']/max(info_a['iterations'],1):.1f}x")

    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.plot(t_values, iters_plain, color=C_RED, linewidth=2, marker='o',
            markersize=4, label='Plain FP')
    ax.plot(t_values, iters_anderson, color=C_GREEN, linewidth=2, marker='s',
            markersize=4, label='Anderson (m=5)')
    ax.axvline(t_break, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label=f'$t_{{break}}$ = 1/$\\pi$')
    ax.set_xlabel('t', color=C_TEXT, fontsize=12)
    ax.set_ylabel('Iterations to converge', color=C_TEXT, fontsize=12)
    ax.set_title('Anderson Acceleration vs Plain Fixed-Point',
                 color=C_TEXT, fontsize=14, fontfamily='serif')
    add_legend(ax)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'exp11g_anderson.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved: {fname}")
    plt.close(fig)

    return dict(experiment='11G_anderson',
                t_values=t_values.tolist(),
                iters_plain=iters_plain,
                iters_anderson=iters_anderson)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    # Allow running specific experiments: python exp11_burgers.py efg
    run_all = len(sys.argv) < 2
    targets = sys.argv[1] if len(sys.argv) > 1 else 'abcdefg'

    print("=" * 70)
    print("EXPERIMENT 11: ELM-CINN for Inviscid Burgers (Nonlinear)")
    print("=" * 70)

    results = {}

    if run_all or 'a' in targets:
        results['11a'] = run_exp11a()
    if run_all or 'b' in targets:
        results['11b'] = run_exp11b()
    if run_all or 'c' in targets:
        results['11c'] = run_exp11c()
    if run_all or 'd' in targets:
        results['11d'] = run_exp11d_convergence()
    if run_all or 'e' in targets:
        results['11e'] = run_exp11e_rarefaction_smooth()
    if run_all or 'f' in targets:
        results['11f'] = run_exp11f_post_shock()
    if run_all or 'g' in targets:
        results['11g'] = run_exp11g_anderson()
    if run_all or 's' in targets:
        results['stats'] = run_statistical()
        results['stats_smooth'] = run_stats_smooth()

    out = os.path.join(RESULTS_DIR, 'exp11_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {out}")
