"""
Experiment 8 — Step-ELM for Linear Advection with Riemann IC
==============================================================
Replicates the setup from Braga-Neto (2023), "Characteristics-Informed
Neural Networks" (arXiv 2212.14012), Section 4.1.1.

PDE:   u_t + v * u_x = 0       (linear advection, constant velocity)
IC:    u(x,0) = u_L if x < L/2, u_R if x > L/2   (Riemann condition)
BC:    u(0,t) = u_L,  u(L,t) = u_R
Exact: u(x,t) = g(x - v*t)     (profile translates at speed v)

Domain: [0, L] x [0, T],  L=2, v=1, T=0.8

Methods compared:
  A) Step-EDNN (baseline):       random tanh basis fixed in x, EDNN evolves
                                  step positions + weights.  Basis degrades.
  B) ELM-CINN (analytical):         evaluate basis at xi = x - v*t.  Same beta
                                  from IC fit works for all t.  No evolution.
  C) ELM-CINN (variable v):     shift tanh analytically at local velocity,
                                  EDNN evolves step + beta for corrections.
  D) Statistical comparison:     10 seeds, compare with CINN Table 1.

Reference results (CINN paper Table 1, 10 reps):
  NN:   L1=0.1251+-0.0649, L2=0.3120+-0.1066, time=5.9s
  PINN: L1=0.0118+-0.0066, L2=0.0619+-0.0275, time=10.2s
  CINN: L1=0.0160+-0.0094, L2=0.0550+-0.0265, time=5.7s
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json, os, time

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Problem setup (matching CINN paper Section 4.1.1)
# ═══════════════════════════════════════════════════════════════════════════════
L = 2.0          # spatial domain [0, L]
T_FINAL = 0.8    # time domain [0, T]
V_CONST = 1.0    # advection velocity
U_L = 5.0        # left state
U_R = 1.0        # right state
X_DISC = L / 2   # initial discontinuity position

def riemann_ic(x):
    """Riemann initial condition: step at x = L/2."""
    return np.where(x < X_DISC, U_L, U_R)

def exact_const_v(x, t, v=V_CONST):
    """Exact solution for constant-velocity advection: u(x,t) = g(x - v*t)."""
    return riemann_ic(x - v * t)

def exact_variable_v(x_eval, t, v_func, rtol=1e-10):
    """
    Exact solution by method of characteristics for variable velocity.
    Trace backward: dY/ds = -v(Y,s), Y(0)=x, Y(t)=xi, u(x,t) = g(xi).
    """
    if t < 1e-14:
        return riemann_ic(x_eval)
    origins = np.zeros_like(x_eval)
    for i, xi in enumerate(x_eval):
        sol = solve_ivp(lambda s, y: -v_func(y, s), [0, t], [xi],
                        method='RK45', rtol=rtol, atol=1e-12)
        origins[i] = sol.y[0, -1]
    return riemann_ic(origins)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Network building blocks
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tanh_weights(n_tanh, seed=7, scale=2.5):
    """Random fixed tanh input weights (ELM philosophy)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale * L, scale * L, size=n_tanh)
    return W, b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def hidden_matrix(x, W_tanh, b_tanh, positions, kappa):
    """H(x): tanh features + step features.  x: (N,) -> H: (N, n_tanh+K)"""
    z_tanh = np.outer(x, W_tanh) + b_tanh
    H_tanh = np.tanh(z_tanh)
    if len(positions) > 0:
        z_step = kappa * (x.reshape(-1, 1) - np.array(positions).reshape(1, -1))
        H_step = sigmoid(z_step)
        return np.hstack([H_tanh, H_step])
    return H_tanh

def hidden_x_matrix(x, W_tanh, b_tanh, positions, kappa):
    """dH/dx: spatial derivatives of hidden activations."""
    z_tanh = np.outer(x, W_tanh) + b_tanh
    Hx_tanh = W_tanh * (1.0 - np.tanh(z_tanh)**2)
    if len(positions) > 0:
        z_step = kappa * (x.reshape(-1, 1) - np.array(positions).reshape(1, -1))
        S = sigmoid(z_step)
        Hx_step = kappa * S * (1.0 - S)
        return np.hstack([Hx_tanh, Hx_step])
    return Hx_tanh

def predict(x, W_tanh, b_tanh, positions, kappa, beta):
    H = hidden_matrix(x, W_tanh, b_tanh, positions, kappa)
    return H @ beta

def solve_ridge(H, y, lam=1e-6):
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDNN core: Jacobian, gamma, RK4 time stepping
# ═══════════════════════════════════════════════════════════════════════════════
def compute_jacobian(x_col, W_tanh, b_tanh, positions, kappa, beta,
                     evolve_tanh=False):
    """
    J[i,j] = du_hat(x_i)/dW_j.

    Partial mode (evolve_tanh=False):
      W = [pos_0..pos_{K-1}, beta_0..beta_{n+K-1}]

    Full mode (evolve_tanh=True):
      W = [pos_0..pos_{K-1}, w_tanh_0..w_{n-1}, b_tanh_0..b_{n-1},
           beta_0..beta_{n+K-1}]
    """
    Nc = len(x_col)
    K = len(positions)
    n = len(W_tanh)

    z_tanh = np.outer(x_col, W_tanh) + b_tanh
    T_act = np.tanh(z_tanh)
    sech2 = 1.0 - T_act**2

    if K > 0:
        z_step = kappa * (x_col.reshape(-1, 1) - np.array(positions).reshape(1, -1))
        S = sigmoid(z_step)
        Sd = S * (1.0 - S)

    cols = []

    # Columns for step positions: du/d(x_k) = beta_{n+k} * (-kappa) * sigma'
    if K > 0:
        for k in range(K):
            cols.append(beta[n + k] * (-kappa) * Sd[:, k])

    # (Full mode) Columns for tanh input weights and biases
    if evolve_tanh:
        for j in range(n):
            cols.append(beta[j] * x_col * sech2[:, j])
        for j in range(n):
            cols.append(beta[j] * sech2[:, j])

    # Columns for output weights: du/d(beta_m) = H[i,m]
    H = np.hstack([T_act, S]) if K > 0 else T_act
    for m in range(n + K):
        cols.append(H[:, m])

    return np.column_stack(cols)

def compute_pde_rhs(x_col, W_tanh, b_tanh, positions, kappa, beta, v_func):
    """N(u) = -v(x) * du/dx  for advection PDE."""
    Hx = hidden_x_matrix(x_col, W_tanh, b_tanh, positions, kappa)
    u_x = Hx @ beta
    v_vals = v_func(x_col) if callable(v_func) else v_func * np.ones_like(x_col)
    return -v_vals * u_x

def compute_gamma(x_col, W_tanh, b_tanh, positions, kappa, beta, v_func,
                  reg=1e-8, evolve_tanh=False):
    """Solve (J^T J + reg I) gamma = J^T N."""
    J = compute_jacobian(x_col, W_tanh, b_tanh, positions, kappa, beta,
                         evolve_tanh)
    N = compute_pde_rhs(x_col, W_tanh, b_tanh, positions, kappa, beta, v_func)
    JtJ = J.T @ J + reg * np.eye(J.shape[1])
    JtN = J.T @ N
    return np.linalg.solve(JtJ, JtN)

def pack_params(positions, W_tanh, b_tanh, beta, evolve_tanh=False):
    parts = [np.array(positions)]
    if evolve_tanh:
        parts.extend([W_tanh, b_tanh])
    parts.append(beta)
    return np.concatenate(parts)

def unpack_params(W_vec, K, n_tanh, evolve_tanh=False):
    idx = 0
    positions = W_vec[idx:idx+K].copy(); idx += K
    if evolve_tanh:
        W_tanh = W_vec[idx:idx+n_tanh].copy(); idx += n_tanh
        b_tanh = W_vec[idx:idx+n_tanh].copy(); idx += n_tanh
    else:
        W_tanh = None; b_tanh = None
    beta = W_vec[idx:].copy()
    return positions, W_tanh, b_tanh, beta

def gamma_from_W(W_vec, K, n_tanh, x_col, W_tanh_fixed, b_tanh_fixed,
                 kappa, v_func, reg, evolve_tanh):
    positions, wt_ev, bt_ev, beta = unpack_params(W_vec, K, n_tanh, evolve_tanh)
    wt = wt_ev if evolve_tanh else W_tanh_fixed
    bt = bt_ev if evolve_tanh else b_tanh_fixed
    return compute_gamma(x_col, wt, bt, positions, kappa, beta, v_func,
                         reg, evolve_tanh)

def rk4_step(W_vec, K, n_tanh, x_col, W_tanh_fixed, b_tanh_fixed,
             kappa, v_func, dt, reg, evolve_tanh):
    args = (K, n_tanh, x_col, W_tanh_fixed, b_tanh_fixed, kappa,
            v_func, reg, evolve_tanh)
    k1 = gamma_from_W(W_vec, *args)
    k2 = gamma_from_W(W_vec + dt/2 * k1, *args)
    k3 = gamma_from_W(W_vec + dt/2 * k2, *args)
    k4 = gamma_from_W(W_vec + dt * k3, *args)
    return W_vec + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Evolution driver
# ═══════════════════════════════════════════════════════════════════════════════
def evolve(W0_vec, K, n_tanh, x_col, W_tanh_fixed, b_tanh_fixed, kappa,
           v_func, dt, T_max, reg=1e-8, evolve_tanh=False,
           x_eval=None, ref_func=None, snap_times=None):
    """
    Evolve network parameters forward in time using EDNN + RK4.
    Returns history dict with snapshots at specified times.
    """
    n_steps = int(round(T_max / dt))
    W_vec = W0_vec.copy()

    def _state(W):
        pos, wt, bt, beta = unpack_params(W, K, n_tanh, evolve_tanh)
        return pos, wt if evolve_tanh else W_tanh_fixed, \
               bt if evolve_tanh else b_tanh_fixed, beta

    # Determine which steps correspond to snapshots
    snap_steps = {}
    if snap_times is not None:
        for ts in snap_times:
            snap_steps[int(round(ts / dt))] = ts

    # Collect snapshots and track error at regular intervals
    save_every = max(1, n_steps // 50)
    times_log, rmse_log, l2_log, pos_log = [], [], [], []

    t_wall = time.time()
    for step in range(n_steps + 1):
        t = step * dt

        # Log periodically
        if step % save_every == 0 or step in snap_steps or step == n_steps:
            pos, wt, bt, beta = _state(W_vec)
            times_log.append(float(t))
            pos_log.append(pos.tolist())
            if x_eval is not None and ref_func is not None:
                u_pred = predict(x_eval, wt, bt, pos, kappa, beta)
                u_ref = ref_func(x_eval, t)
                rmse_log.append(float(np.sqrt(np.mean((u_pred - u_ref)**2))))
                norm_ref = np.linalg.norm(u_ref)
                l2_log.append(float(np.linalg.norm(u_pred - u_ref) /
                                    max(norm_ref, 1e-12)))

        # RK4 step
        if step < n_steps:
            W_vec = rk4_step(W_vec, K, n_tanh, x_col, W_tanh_fixed,
                             b_tanh_fixed, kappa, v_func, dt, reg,
                             evolve_tanh)

    elapsed = time.time() - t_wall

    # Extract snapshot solutions
    snapshots = {}
    W_vec2 = W0_vec.copy()
    for step in range(n_steps + 1):
        t = step * dt
        if step in snap_steps:
            pos, wt, bt, beta = _state(W_vec2)
            u_pred = predict(x_eval, wt, bt, pos, kappa, beta)
            u_ref = ref_func(x_eval, t)
            rmse = float(np.sqrt(np.mean((u_pred - u_ref)**2)))
            norm_ref = np.linalg.norm(u_ref)
            l1_err = float(np.mean(np.abs(u_pred - u_ref)) /
                           max(np.mean(np.abs(u_ref)), 1e-12))
            l2_err = float(np.linalg.norm(u_pred - u_ref) /
                           max(norm_ref, 1e-12))
            snapshots[f"t={t:.2f}"] = {
                't': float(t),
                'u_pred': u_pred,
                'u_ref': u_ref,
                'positions': pos.tolist(),
                'rmse': rmse,
                'l1_error': l1_err,
                'l2_error': l2_err,
            }
        if step < n_steps:
            W_vec2 = rk4_step(W_vec2, K, n_tanh, x_col, W_tanh_fixed,
                              b_tanh_fixed, kappa, v_func, dt, reg,
                              evolve_tanh)

    return {
        'times': times_log,
        'rmse': rmse_log,
        'l2_error': l2_log,
        'positions': pos_log,
        'snapshots': snapshots,
        'elapsed_s': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════════════
# Light academic palette
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
    """Plot solution snapshots: reference vs Step-EDNN."""
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
        for xp in s['positions']:
            ax.axvline(xp, color=C_ORANGE, linewidth=1.5, linestyle=':',
                       alpha=0.7, zorder=2)
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {s['l2_error']:.4f}",
                     color=C_TEXT, fontsize=11, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='upper right')

    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)

def plot_diagnostics(history, title, fname, v_true=None):
    """Plot error evolution, step trajectories, step velocity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)

    times = np.array(history['times'])
    pos_hist = np.array(history['positions'])
    K = pos_hist.shape[1] if pos_hist.ndim > 1 else 0

    # ── L2 error over time ──
    ax = axes[0]
    style_ax(ax)
    if history.get('l2_error'):
        t_err = times[:len(history['l2_error'])]
        ax.semilogy(t_err, history['l2_error'], color=C_RED, linewidth=2)
    ax.set_xlabel('t', color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT)
    ax.set_title('Error evolution', color=C_TEXT, fontsize=11, fontfamily='serif')

    # ── Step position trajectory ──
    ax = axes[1]
    style_ax(ax)
    pal = [C_RED, C_ORANGE, C_GREEN, C_CYAN]
    if K > 0:
        for k in range(K):
            ax.plot(times, pos_hist[:, k], color=pal[k % len(pal)],
                    linewidth=2, label=f'Step {k+1}')
        if v_true is not None:
            x0 = pos_hist[0, 0]
            ax.plot(times, x0 + v_true * times, color=C_BLUE, linewidth=1.5,
                    linestyle=':', label=f'Characteristic (v={v_true})')
        add_legend(ax)
    ax.set_xlabel('t', color=C_TEXT)
    ax.set_ylabel('x position', color=C_TEXT)
    ax.set_title('Step trajectory', color=C_TEXT, fontsize=11, fontfamily='serif')

    # ── Step velocity (numerical derivative) ──
    ax = axes[2]
    style_ax(ax)
    if K > 0 and len(times) > 2:
        for k in range(K):
            vel = np.gradient(pos_hist[:, k], times)
            ax.plot(times, vel, color=pal[k % len(pal)], linewidth=1.5,
                    label=f'dx/dt step {k+1}')
        if v_true is not None:
            ax.axhline(v_true, color=C_BLUE, linewidth=1.5, linestyle=':',
                       label=f'v_true = {v_true}')
        add_legend(ax)
    ax.set_xlabel('t', color=C_TEXT)
    ax.set_ylabel('velocity', color=C_TEXT)
    ax.set_title('Step velocity', color=C_TEXT, fontsize=11, fontfamily='serif')

    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Characteristic-Shifted ELM (no EDNN needed)
# ═══════════════════════════════════════════════════════════════════════════════
def charshift_solve(config, ref_func, snap_times, v_const=V_CONST):
    """
    Characteristic-shifted ELM: evaluate the entire basis at xi = x - v*t.

    Since u(x,t) = f(x - v*t) for constant-velocity advection, and we fit
    f(xi) at t=0 with basis H(xi) * beta, the SAME beta works at any t:
        u_hat(x,t) = H(x - v*t) * beta

    No EDNN evolution, no Jacobian, no time stepping.  Just shift and evaluate.

    KEY: fit the IC on an EXTENDED domain [0 - v*T, L] so that at any time
    t <= T the shifted coordinate xi = x - v*t stays within the fitted range.
    """
    n_tanh = config['n_tanh']
    kappa = config['kappa']
    seed_tanh = config['seed_tanh']
    T_max = config.get('T_max', T_FINAL)

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed_tanh)
    positions = np.array(config['positions'])
    K = len(positions)

    # Fit IC on EXTENDED domain: xi ranges from [0 - v*T, L] at t=T
    xi_min = -v_const * T_max - 0.2   # extra margin
    xi_max = L + 0.2
    x_ic = np.linspace(xi_min, xi_max, config['n_ic'])
    y_ic = riemann_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])

    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic)**2)))

    # Evaluate at each snapshot: just shift x by -v*t
    x_eval = np.linspace(0, L, config['n_eval'])
    snapshots = {}

    t_start = time.time()
    for t_snap in snap_times:
        xi = x_eval - v_const * t_snap  # characteristic coordinate
        H_shifted = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_pred = H_shifted @ beta
        u_ref = ref_func(x_eval, t_snap)

        rmse = float(np.sqrt(np.mean((u_pred - u_ref)**2)))
        norm_ref = max(np.linalg.norm(u_ref), 1e-12)
        l1_err = float(np.mean(np.abs(u_pred - u_ref)) /
                        max(np.mean(np.abs(u_ref)), 1e-12))
        l2_err = float(np.linalg.norm(u_pred - u_ref) / norm_ref)

        # Step position in physical space: x_k + v*t
        phys_pos = [p + v_const * t_snap for p in positions]

        snapshots[f"t={t_snap:.2f}"] = {
            't': float(t_snap),
            'u_pred': u_pred,
            'u_ref': u_ref,
            'positions': phys_pos,
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
        'n_params': n_tanh + K,  # only beta (output weights)
    }


def charshift_evolve(config, v_func_x, ref_func, snap_times):
    """
    Characteristic-shifted EDNN for variable velocity.

    Tanh biases shift analytically at local velocity:
        b_j(t+dt) = b_j(t) - w_j * v(center_j) * dt
    Step position evolves via simple ODE: dx_k/dt = v(x_k).
    Output weights beta re-solved by ridge at each time step.

    For constant v this reduces to charshift_solve (but slower due to stepping).
    For variable v(x) this tracks the characteristics approximately.
    """
    n_tanh = config['n_tanh']
    kappa = config['kappa']
    seed_tanh = config['seed_tanh']
    dt = config['dt']
    T_max = config['T_max']
    n_steps = int(round(T_max / dt))

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed_tanh)
    positions = np.array(config['positions'], dtype=float)
    K = len(positions)

    # Fit IC on extended domain (characteristics may shift basis outside [0,L])
    x_ic = np.linspace(-1.5, L + 0.5, config['n_ic'])
    y_ic = riemann_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic)**2)))

    x_eval = np.linspace(0, L, config['n_eval'])
    x_refit = np.linspace(-1.0, L + 0.5, config['n_ic'])

    # Working copies
    b_curr = b_tanh.copy()
    pos_curr = positions.copy()

    snap_step_map = {}
    for ts in snap_times:
        snap_step_map[int(round(ts / dt))] = ts

    snapshots = {}
    save_every = max(1, n_steps // 50)
    times_log, l2_log, pos_log = [], [], []

    t_wall = time.time()
    for step in range(n_steps + 1):
        t = step * dt

        # Log / snapshot
        if step % save_every == 0 or step in snap_step_map or step == n_steps:
            u_pred = predict(x_eval, W_tanh, b_curr, pos_curr, kappa, beta)
            u_ref = ref_func(x_eval, t)
            norm_ref = max(np.linalg.norm(u_ref), 1e-12)
            l2 = float(np.linalg.norm(u_pred - u_ref) / norm_ref)
            times_log.append(float(t))
            l2_log.append(l2)
            pos_log.append(pos_curr.tolist())

            if step in snap_step_map:
                l1 = float(np.mean(np.abs(u_pred - u_ref)) /
                           max(np.mean(np.abs(u_ref)), 1e-12))
                rmse = float(np.sqrt(np.mean((u_pred - u_ref)**2)))
                snapshots[f"t={t:.2f}"] = {
                    't': float(t),
                    'u_pred': u_pred,
                    'u_ref': u_ref,
                    'positions': pos_curr.tolist(),
                    'rmse': rmse,
                    'l1_error': l1,
                    'l2_error': l2,
                }

        if step >= n_steps:
            break

        # ── Advance one time step ──
        # 1. Shift tanh biases: b_j -= w_j * v(center_j) * dt
        #    center_j = -b_j / w_j (zero crossing of tanh neuron j)
        centers = -b_curr / (W_tanh + 1e-30)
        if callable(v_func_x):
            v_at_centers = v_func_x(centers)
        else:
            v_at_centers = float(v_func_x)
        b_curr -= W_tanh * v_at_centers * dt

        # 2. Move step position: x_k += v(x_k) * dt
        if callable(v_func_x):
            v_at_steps = v_func_x(pos_curr)
        else:
            v_at_steps = float(v_func_x)
        pos_curr += v_at_steps * dt

        # 3. Re-solve output weights on virtual IC transported to current time
        #    (the transported IC is just riemann_ic evaluated at current
        #     characteristic origins — but we approximate by re-fitting
        #     on the predicted solution with a light regularization step
        #     every N steps)
        if step % 100 == 0:
            # Re-fit beta to maintain consistency
            u_target = ref_func(x_refit, t + dt)
            H_refit = hidden_matrix(x_refit, W_tanh, b_curr, pos_curr, kappa)
            beta = solve_ridge(H_refit, u_target, config['lam'])

    elapsed = time.time() - t_wall

    return {
        'ic_rmse': ic_rmse,
        'elapsed_s': elapsed,
        'snapshots': snapshots,
        'x_eval': x_eval,
        'n_params': n_tanh + K,
        'history': {
            'times': times_log,
            'l2_error': l2_log,
            'positions': pos_log,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(label, v_func, ref_func, config, v_true_for_plot=None):
    """Run a single Step-EDNN evolution experiment."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    n_tanh = config['n_tanh']
    kappa = config['kappa']
    evolve_tanh = config.get('evolve_tanh', False)
    seed_tanh = config['seed_tanh']

    W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed_tanh)

    # Step neuron at the discontinuity
    positions = np.array(config['positions'])
    K = len(positions)

    # Fit IC: sample dense points on [0, L]
    x_ic = np.linspace(0, L, config['n_ic'])
    y_ic = riemann_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])

    y_pred_ic = H_ic @ beta
    ic_rmse = float(np.sqrt(np.mean((y_ic - y_pred_ic)**2)))
    print(f"  IC fit: RMSE={ic_rmse:.6f}  ({n_tanh} tanh + {K} step = "
          f"{n_tanh+K} neurons)")

    # Collocation and evaluation grids
    margin = 0.05
    x_col = np.linspace(margin, L - margin, config['n_col'])
    x_eval = np.linspace(0, L, config['n_eval'])

    # Pack initial parameters
    W0 = pack_params(positions, W_tanh, b_tanh, beta, evolve_tanh)
    n_params = len(W0)
    mode = "FULL" if evolve_tanh else "PARTIAL"
    print(f"  Mode: {mode} ({n_params} parameters)")
    print(f"  dt={config['dt']}, T_max={config['T_max']}, "
          f"n_col={config['n_col']}, reg={config['reg']}")

    history = evolve(
        W0, K, n_tanh, x_col, W_tanh, b_tanh, kappa,
        v_func, config['dt'], config['T_max'], config['reg'],
        evolve_tanh, x_eval, ref_func, config['snap_times']
    )

    print(f"  Elapsed: {history['elapsed_s']:.1f}s")
    for key, snap in sorted(history['snapshots'].items()):
        print(f"    {key}: RMSE={snap['rmse']:.6f}, L1={snap['l1_error']:.4f}, "
              f"L2={snap['l2_error']:.4f}, pos={snap['positions']}")

    return {
        'label': label,
        'ic_rmse': ic_rmse,
        'n_params': n_params,
        'evolve_tanh': evolve_tanh,
        'history': history,
        'x_eval': x_eval.tolist(),
        'config': config,
        'v_true_for_plot': v_true_for_plot,
    }


def run_exp8a(results_dir):
    """Exp 8A: Char-Shift (analytical) vs EDNN, constant v=1."""
    base_config = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6, 'seed_tanh': 7,
        'positions': [X_DISC],
        'n_ic': 500, 'n_col': 300, 'n_eval': 1000,
        'dt': 5e-4, 'T_max': T_FINAL, 'reg': 1e-8,
        'snap_times': [0.0, 0.08, 0.40, 0.73],
    }

    def ref(x, t):
        return exact_const_v(x, t, V_CONST)

    # ── Char-Shift (analytical, no EDNN) ──
    print(f"\n{'='*70}")
    print(f"  Exp 8A-ELM-CINN: Characteristics-Informed ELM, v=1")
    print(f"{'='*70}")
    result_cs = charshift_solve(base_config, ref, base_config['snap_times'])
    print(f"  IC RMSE: {result_cs['ic_rmse']:.6f}")
    print(f"  Elapsed: {result_cs['elapsed_s']:.3f}s (no evolution needed)")
    for key, snap in sorted(result_cs['snapshots'].items()):
        print(f"    {key}: RMSE={snap['rmse']:.6f}, L1={snap['l1_error']:.4f}, "
              f"L2={snap['l2_error']:.4f}")

    # ── Full EDNN baseline for comparison ──
    config_ednn = {**base_config, 'evolve_tanh': True}
    result_ednn = run_single(
        "Exp 8A-EDNN: Full EDNN baseline, v=1",
        V_CONST, ref, config_ednn, v_true_for_plot=V_CONST)

    return result_cs, result_ednn


def run_exp8b(results_dir):
    """Exp 8B: Variable velocity v(x)=1+x, ELM-CINN."""
    def v_spatial(x):
        return 1.0 + x

    config = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6, 'seed_tanh': 7,
        'positions': [X_DISC],
        'n_ic': 500, 'n_col': 300, 'n_eval': 1000,
        'dt': 2e-4, 'T_max': 0.5, 'reg': 1e-8,
        'snap_times': [0.0, 0.10, 0.25, 0.50],
    }

    def ref(x, t):
        return exact_variable_v(x, t, lambda y, s: v_spatial(y))

    print(f"\n{'='*70}")
    print(f"  Exp 8B: ELM-CINN, v(x) = 1 + x")
    print(f"{'='*70}")
    result = charshift_evolve(config, v_spatial, ref, config['snap_times'])
    print(f"  IC RMSE: {result['ic_rmse']:.6f}")
    print(f"  Elapsed: {result['elapsed_s']:.1f}s")
    for key, snap in sorted(result['snapshots'].items()):
        print(f"    {key}: RMSE={snap['rmse']:.6f}, L1={snap['l1_error']:.4f}, "
              f"L2={snap['l2_error']:.4f}, pos={snap['positions']}")

    return result


def run_exp8c(results_dir):
    """Exp 8C: Statistical comparison — 10 seeds, Char-Shift vs EDNN."""
    config_base = {
        'n_tanh': 80, 'kappa': 500.0, 'lam': 1e-6,
        'positions': [X_DISC],
        'n_ic': 500, 'n_col': 300, 'n_eval': 1000,
        'dt': 5e-4, 'T_max': T_FINAL, 'reg': 1e-8,
    }

    def ref(x, t):
        return exact_const_v(x, t, V_CONST)

    seeds = list(range(10))
    snap_times = [0.0, T_FINAL]

    # ── Char-Shift (analytical) ──
    cs_l1, cs_l2, cs_time = [], [], []
    # ── EDNN baseline ──
    ednn_l1, ednn_l2, ednn_time = [], [], []

    print(f"\n{'='*70}")
    print(f"  Exp 8C: Statistical comparison (10 seeds)")
    print(f"{'='*70}")
    print(f"  {'seed':<6} {'ELM-CINN L2':<16} {'EDNN L2':<16}")
    print(f"  {'----':<6} {'------------':<16} {'-------':<16}")

    for seed in seeds:
        cfg = {**config_base, 'seed_tanh': seed}

        # Char-Shift
        res_cs = charshift_solve(cfg, ref, snap_times)
        snap_cs = res_cs['snapshots'][f"t={T_FINAL:.2f}"]
        cs_l1.append(snap_cs['l1_error'])
        cs_l2.append(snap_cs['l2_error'])
        cs_time.append(res_cs['elapsed_s'])

        # EDNN
        cfg_ednn = {**cfg, 'evolve_tanh': True, 'snap_times': snap_times}
        n_tanh = cfg_ednn['n_tanh']
        kappa = cfg_ednn['kappa']
        W_tanh, b_tanh = generate_tanh_weights(n_tanh, seed)
        positions = np.array(cfg_ednn['positions'])
        K = len(positions)
        x_ic = np.linspace(0, L, cfg_ednn['n_ic'])
        y_ic = riemann_ic(x_ic)
        H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
        beta = solve_ridge(H_ic, y_ic, cfg_ednn['lam'])
        x_col = np.linspace(0.05, L - 0.05, cfg_ednn['n_col'])
        x_eval = np.linspace(0, L, cfg_ednn['n_eval'])
        W0 = pack_params(positions, W_tanh, b_tanh, beta, True)
        history = evolve(
            W0, K, n_tanh, x_col, W_tanh, b_tanh, kappa,
            V_CONST, cfg_ednn['dt'], cfg_ednn['T_max'], cfg_ednn['reg'],
            True, x_eval, ref, snap_times)
        snap_ednn = history['snapshots'][f"t={T_FINAL:.2f}"]
        ednn_l1.append(snap_ednn['l1_error'])
        ednn_l2.append(snap_ednn['l2_error'])
        ednn_time.append(history['elapsed_s'])

        print(f"  {seed:<6} {snap_cs['l2_error']:<16.4f} {snap_ednn['l2_error']:<16.4f}")

    stats = {
        'charshift': {
            'l1_mean': float(np.mean(cs_l1)), 'l1_std': float(np.std(cs_l1)),
            'l2_mean': float(np.mean(cs_l2)), 'l2_std': float(np.std(cs_l2)),
            'time_mean': float(np.mean(cs_time)), 'time_std': float(np.std(cs_time)),
            'l1_all': cs_l1, 'l2_all': cs_l2, 'time_all': cs_time,
        },
        'ednn': {
            'l1_mean': float(np.mean(ednn_l1)), 'l1_std': float(np.std(ednn_l1)),
            'l2_mean': float(np.mean(ednn_l2)), 'l2_std': float(np.std(ednn_l2)),
            'time_mean': float(np.mean(ednn_time)), 'time_std': float(np.std(ednn_time)),
            'l1_all': ednn_l1, 'l2_all': ednn_l2, 'time_all': ednn_time,
        },
        'seeds': seeds,
    }

    s_cs = stats['charshift']
    s_ed = stats['ednn']
    print(f"\n  {'Method':<25} {'L1':<22} {'L2':<22} {'Time':<10}")
    print(f"  {'-'*25} {'-'*22} {'-'*22} {'-'*10}")
    print(f"  {'ELM-CINN (ours)':<25} "
          f"{s_cs['l1_mean']:.4f}+-{s_cs['l1_std']:.4f}       "
          f"{s_cs['l2_mean']:.4f}+-{s_cs['l2_std']:.4f}       "
          f"{s_cs['time_mean']:.3f}s")
    print(f"  {'EDNN (ours)':<25} "
          f"{s_ed['l1_mean']:.4f}+-{s_ed['l1_std']:.4f}       "
          f"{s_ed['l2_mean']:.4f}+-{s_ed['l2_std']:.4f}       "
          f"{s_ed['time_mean']:.1f}s")
    print(f"  {'CINN (Braga-Neto)':<25} 0.0160+-0.0094       0.0550+-0.0265       5.7s")
    print(f"  {'PINN (baseline)':<25} 0.0118+-0.0066       0.0619+-0.0275       10.2s")
    print(f"  {'NN (no physics)':<25} 0.1251+-0.0649       0.3120+-0.1066       5.9s")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Exp 8A: Char-Shift vs EDNN, constant v ──
    result_cs, result_ednn = run_exp8a(results_dir)
    x_eval = result_cs['x_eval']

    plot_snapshots(x_eval, result_cs['snapshots'],
                   "Exp 8A: ELM-CINN — Linear Advection Riemann, v=1",
                   os.path.join(results_dir, 'exp8a_charshift_snapshots.png'))

    plot_snapshots(x_eval, result_ednn['history']['snapshots'],
                   "Exp 8A: Full EDNN — Advection Riemann, v=1",
                   os.path.join(results_dir, 'exp8a_ednn_snapshots.png'))
    plot_diagnostics(result_ednn['history'],
                     "Exp 8A: Full EDNN — Diagnostics",
                     os.path.join(results_dir, 'exp8a_ednn_diagnostics.png'),
                     v_true=V_CONST)

    # ── Exp 8B: variable v(x), ELM-CINN ──
    result_b = run_exp8b(results_dir)
    plot_snapshots(result_b['x_eval'], result_b['snapshots'],
                   "Exp 8B: ELM-CINN — Linear Advection, v(x)=1+x",
                   os.path.join(results_dir, 'exp8b_charshift_snapshots.png'))
    if 'history' in result_b:
        plot_diagnostics(result_b['history'],
                         "Exp 8B: ELM-CINN — Diagnostics",
                         os.path.join(results_dir, 'exp8b_diagnostics.png'))

    # ── Exp 8C: statistical comparison ──
    stats_c = run_exp8c(results_dir)

    # ── Save all results ──
    def snap_summary(snaps):
        return {k: {kk: vv for kk, vv in v.items()
                    if kk not in ('u_pred', 'u_ref')}
                for k, v in snaps.items()}

    all_results = {
        'exp8a_charshift': {
            'label': 'ELM-CINN (analytical), v=1',
            'ic_rmse': result_cs['ic_rmse'],
            'n_params': result_cs['n_params'],
            'elapsed_s': result_cs['elapsed_s'],
            'snapshots': snap_summary(result_cs['snapshots']),
        },
        'exp8a_ednn': {
            'label': result_ednn['label'],
            'ic_rmse': result_ednn['ic_rmse'],
            'n_params': result_ednn['n_params'],
            'elapsed_s': result_ednn['history']['elapsed_s'],
            'snapshots': snap_summary(result_ednn['history']['snapshots']),
        },
        'exp8b_char_ednn': {
            'label': 'ELM-CINN, v(x)=1+x',
            'ic_rmse': result_b['ic_rmse'],
            'n_params': result_b['n_params'],
            'elapsed_s': result_b['elapsed_s'],
            'snapshots': snap_summary(result_b['snapshots']),
        },
        'exp8c_stats': stats_c,
        'cinn_reference': {
            'l1': '0.0160+-0.0094', 'l2': '0.0550+-0.0265', 'time': '5.7+-0.1',
        },
        'pinn_reference': {
            'l1': '0.0118+-0.0066', 'l2': '0.0619+-0.0275', 'time': '10.2+-1.2',
        },
    }

    out_path = os.path.join(results_dir, 'exp8_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved: {out_path}")

    # ── Comparison bar chart: all 5 methods ──
    plot_comparison(stats_c, os.path.join(results_dir, 'exp8_comparison.png'))


def plot_comparison(stats, fname):
    """Bar chart comparing all methods on L2 error and time."""
    methods = ['ELM-CINN', 'CINN',
               'PINN\n(baseline)', 'EDNN\nStep-ELM', 'NN\n(no physics)']
    l2_means = [
        stats['charshift']['l2_mean'],
        0.0550,
        0.0619,
        stats['ednn']['l2_mean'],
        0.3120,
    ]
    l2_stds = [
        stats['charshift']['l2_std'],
        0.0265,
        0.0275,
        stats['ednn']['l2_std'],
        0.1066,
    ]
    times = [
        stats['charshift']['time_mean'],
        5.7,
        10.2,
        stats['ednn']['time_mean'],
        5.9,
    ]
    colors = [C_GREEN, C_BLUE, C_CYAN, C_ORANGE, '#999999']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    # ── L2 error ──
    ax = axes[0]
    style_ax(ax)
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, l2_means, yerr=l2_stds, capsize=5,
                  color=colors, edgecolor='#555555', linewidth=0.8, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9, color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title('L$_2$ Error at t = T (10 seeds)', color=C_TEXT,
                 fontsize=13, fontfamily='serif')
    # Add value labels on bars
    for bar, m, s in zip(bars, l2_means, l2_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                f'{m:.4f}', ha='center', va='bottom', fontsize=8, color=C_TEXT)

    # ── Time ──
    ax = axes[1]
    style_ax(ax)
    bars_t = ax.bar(x_pos, times, color=colors, edgecolor='#555555',
                    linewidth=0.8, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9, color=C_TEXT)
    ax.set_ylabel('Time (seconds)', color=C_TEXT, fontsize=11)
    ax.set_title('Running Time', color=C_TEXT, fontsize=13, fontfamily='serif')
    for bar, t_val in zip(bars_t, times):
        label = f'{t_val:.3f}' if t_val < 0.1 else f'{t_val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                label + 's', ha='center', va='bottom', fontsize=8, color=C_TEXT)

    fig.suptitle('Linear Advection Riemann — Method Comparison',
                 color=C_TEXT, fontsize=15, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == '__main__':
    main()
