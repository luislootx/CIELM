"""
Unified CIELM: Single algorithm for pre- and post-shock Burgers
================================================================
No separate modes. One algorithm that automatically:
  1. Solves characteristics via Newton from BOTH boundaries
  2. Detects shocks where left/right solutions disagree
  3. Constructs entropy solution with interpretable shock position

Algorithm:
  LEFT march:  x_min -> x_max  (Newton continuation, ascending)
  RIGHT march: x_max -> x_min  (Newton continuation, descending)

  If LEFT[i] == RIGHT[i]:  smooth solution at x[i]
  If LEFT[i] != RIGHT[i]:  shock between this x and the previous

The shock position is the x where LEFT and RIGHT solutions cross.
The step neuron location is directly readable from this crossing.

PDE: u_t + u*u_x = 0,  u0(x) = -sin(pi*x)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── ELM ──────────────────────────────────────────────────────────────────────

def generate_tanh_weights(n, seed=7, scale=2.5, ds=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, n), rng.uniform(-scale*ds, scale*ds, n)


def fit_elm(x, y, W, b, lam=1e-8):
    H = np.hstack([np.tanh(np.outer(x, W) + b), np.ones((len(x), 1))])
    beta_full = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ y)
    return beta_full[:-1], beta_full[-1]


def elm_eval(x, W, b, beta, bias):
    return np.tanh(np.outer(x, W) + b) @ beta + bias


# ── Newton continuation (core primitive) ─────────────────────────────────────

def newton_march(x_sorted, t, W, b, beta, bias, max_iter=50, tol=1e-13):
    """
    Solve g(xi) = xi + u_ELM(xi)*t - x = 0 via Newton with continuation.
    x_sorted MUST be monotonically ordered.
    Returns (u_values, xi_values) in same order.
    """
    N = len(x_sorted)
    xi_arr = np.empty(N)
    u_arr = np.empty(N)
    beta_W = beta * W

    xi_curr = float(x_sorted[0])

    for i in range(N):
        x_target = float(x_sorted[i])
        for _ in range(max_iter):
            z = W * xi_curr + b
            tanh_z = np.tanh(z)
            u_val = float(np.dot(beta, tanh_z)) + bias
            sech2 = 1.0 - tanh_z * tanh_z
            du_val = float(np.dot(beta_W, sech2))

            g = xi_curr + u_val * t - x_target
            gp = 1.0 + du_val * t
            if abs(gp) < 1e-15:
                gp = 1e-15
            step = g / gp
            xi_curr -= step
            if abs(step) < tol:
                break
        xi_arr[i] = xi_curr
        u_arr[i] = float(np.dot(beta, np.tanh(W * xi_curr + b))) + bias

    return u_arr, xi_arr


# ── Unified solver ───────────────────────────────────────────────────────────

def unified_cielm(x, t, W, b, beta, bias, shock_tol=0.01):
    """
    Single-pass unified CIELM solver.

    1. March Newton from LEFT boundary (ascending x)
    2. March Newton from RIGHT boundary (descending x)
    3. Compare: where they agree => smooth; where they disagree => shock
    4. Return entropy solution + detected shock position (or None)

    Returns: (u_pred, shock_info)
      shock_info = None (no shock) or {'x_shock': float, 'u_L': float, 'u_R': float}
    """
    if t < 1e-14:
        return elm_eval(x, W, b, beta, bias), None

    # LEFT march: ascending x
    u_left, xi_left = newton_march(x, t, W, b, beta, bias)

    # RIGHT march: descending x, then reverse
    u_right_rev, xi_right_rev = newton_march(x[::-1], t, W, b, beta, bias)
    u_right = u_right_rev[::-1]
    xi_right = xi_right_rev[::-1]

    # Compare: find where solutions disagree
    diff = np.abs(u_left - u_right)
    max_diff = np.max(diff)

    if max_diff < shock_tol:
        # No shock detected -- solutions agree everywhere
        u_pred = 0.5 * (u_left + u_right)
        return u_pred, None

    # Shock detected: find position as center of disagreement region.
    # The LEFT march is correct for x < x_shock.
    # The RIGHT march is correct for x > x_shock.
    # Their xi values diverge at the shock: xi_left > xi_right in the
    # shock zone (left branch ξ crosses over right branch ξ).
    # The shock position is where the xi gap opens:
    xi_diff = xi_left - xi_right
    # The shock is where xi_diff transitions from ~0 to large.
    # Use the midpoint of the "gap" region.
    gap_mask = np.abs(xi_diff) > 0.01
    gap_indices = np.where(gap_mask)[0]

    if len(gap_indices) == 0:
        # Fallback: max diff location
        i_shock = int(np.argmax(diff))
    else:
        # Shock is at the CENTER of the gap region
        i_shock = gap_indices[len(gap_indices) // 2]

    x_shock = float(x[i_shock])

    # Refine: in the gap region, the exact shock is where the
    # Rankine-Hugoniot condition is satisfied: s = (u_L + u_R) / 2.
    # For antisymmetric IC, s=0, so x_shock should be near 0.
    # Use weighted average of gap positions by xi_diff magnitude.
    if len(gap_indices) > 1:
        weights = np.abs(xi_diff[gap_indices])
        x_shock = float(np.average(x[gap_indices], weights=weights))

    # Construct entropy solution
    u_pred = np.where(x < x_shock, u_left, u_right)

    # Jump values
    # Find values just outside the gap
    left_edge = gap_indices[0]
    right_edge = gap_indices[-1]
    u_L = float(u_left[max(0, left_edge - 1)])
    u_R = float(u_right[min(len(x)-1, right_edge + 1)])

    return u_pred, {'x_shock': x_shock, 'u_L': u_L, 'u_R': u_R,
                    'jump': abs(u_L - u_R)}


# ── Exact solution ───────────────────────────────────────────────────────────

def exact_colehopf(x, t, nu=0.001):
    if t < 1e-14:
        return -np.sin(np.pi * x)
    nq = max(8000, int(12000 / np.sqrt(nu)))
    xi = np.linspace(-3, 3, nq)
    dxi = xi[1] - xi[0]
    Phi = (1 - np.cos(np.pi * xi)) / np.pi
    d = x[:, None] - xi[None, :]
    e = -(d**2) / (4*nu*t) + Phi[None, :] / (2*nu)
    e -= e.max(axis=1, keepdims=True)
    w = np.exp(e)
    return np.sum((d/t)*w, axis=1)*dxi / (np.sum(w, axis=1)*dxi)


# ── Experiment ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Unified CIELM: Automatic shock detection + entropy solution")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi
    nu = 0.001

    # Fit IC
    n_tanh = 200
    W, b = generate_tanh_weights(n_tanh, 7, scale=3.5, ds=2.5)
    x_ic = np.linspace(-3.0, 3.0, 2000)
    y_ic = -np.sin(np.pi * x_ic)
    beta, bias = fit_elm(x_ic, y_ic, W, b, lam=1e-10)

    rmse = np.sqrt(np.mean((elm_eval(x_ic, W, b, beta, bias) - y_ic)**2))
    print(f"  IC: {n_tanh} neurons, RMSE={rmse:.2e}")
    print(f"  t_break = {t_break:.4f}")

    x_eval = np.linspace(x_min, x_max, 500)

    test_times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  t_break, 0.35, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0]

    print(f"\n  {'t':>6s} | {'L2':>10s} | {'shock':>8s} | {'jump':>6s} | {'x_s':>6s}")
    print("  " + "-" * 55)

    results = []
    for t in test_times:
        u_exact = exact_colehopf(x_eval, t, nu)
        norm = max(np.linalg.norm(u_exact), 1e-12)

        t0 = time.time()
        u_pred, shock_info = unified_cielm(x_eval, t, W, b, beta, bias)
        elapsed = time.time() - t0

        l2 = float(np.linalg.norm(u_pred - u_exact) / norm)

        if shock_info is not None:
            shock_str = f"x={shock_info['x_shock']:+.4f}"
            jump_str = f"{shock_info['jump']:.3f}"
        else:
            shock_str = "none"
            jump_str = "-"

        print(f"  {t:6.4f} | {l2:10.2e} | {shock_str:>8s} | {jump_str:>6s} | ({elapsed:.3f}s)")

        results.append({
            't': float(t), 'l2': l2, 'shock': shock_info,
            'u_pred': u_pred, 'u_exact': u_exact
        })

    # ── Plot ──
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), facecolor='white')
    axes = axes.flatten()

    C_EX = '#2166ac'
    C_PR = '#d6604d'
    C_SH = '#e08214'

    for i, r in enumerate(results):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.set_facecolor('#fafafa')
        ax.grid(True, color='#e0e0e0', lw=0.5, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor('#aaa')

        ax.plot(x_eval, r['u_exact'], color=C_EX, lw=2.5, zorder=3, label='Exact')
        ax.plot(x_eval, r['u_pred'], color=C_PR, lw=1.8, ls='--', zorder=4,
                label=f'CIELM (L2={r["l2"]:.1e})')

        if r['shock'] is not None:
            xs = r['shock']['x_shock']
            ax.axvline(xs, color=C_SH, lw=1.5, ls=':', alpha=0.6)
            ax.plot(xs, 0, 's', color=C_SH, ms=6, zorder=6)
            ax.set_title(f't={r["t"]:.4f}  shock@{xs:+.3f}  '
                         f'jump={r["shock"]["jump"]:.2f}', fontsize=8)
        else:
            ax.set_title(f't={r["t"]:.4f}  [smooth]  L2={r["l2"]:.1e}', fontsize=8)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.4, 1.4)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)

    fig.suptitle('Unified CIELM: Automatic Shock Detection\n'
                 'Single algorithm, no separate pre/post-shock modes',
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(RESULTS_DIR, 'unified_cielm.png')
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out}")

    return results


if __name__ == '__main__':
    main()
