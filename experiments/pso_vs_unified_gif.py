"""
Side-by-side GIF: PSO-PINN (Caio) vs Unified CIELM (ours)
==========================================================
Left:  Frames from Caio's PSO-PINN GIF (viscous Burgers, nu=0.01)
Right: Unified CIELM — automatic pre/post-shock (inviscid, nu->0)

Uses unified_cielm.py which handles the full timeline seamlessly:
  - Pre-shock:  Newton from both boundaries agree -> smooth solution
  - Post-shock: Newton from both boundaries disagree -> shock detected
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io, os, time, gc

from unified_cielm import (
    generate_tanh_weights, fit_elm, elm_eval,
    unified_cielm, exact_colehopf,
    RESULTS_DIR
)

# Colors
C_EXACT = '#2166ac'
C_PRED  = '#d6604d'
C_SHOCK = '#e08214'
C_TEXT  = '#333333'
C_SPINE = '#aaaaaa'


def main():
    # ── Caio's PSO-PINN GIF ──
    caio_path = (r'C:\Users\luisl\OneDrive\Documentos\TAMU\Research'
                 r'\Implementations\PSOPINN\PSO-PINN\examples\images\burgers.gif')
    if not os.path.exists(caio_path):
        print(f"ERROR: Caio's GIF not found: {caio_path}")
        return

    caio_gif = Image.open(caio_path)
    n_frames = caio_gif.n_frames
    print(f"Caio's GIF: {n_frames} frames, {caio_gif.size}")

    # ── Setup unified CIELM ──
    t_break = 1.0 / np.pi
    x_eval = np.linspace(-1.0, 1.0, 500)
    t_values = np.linspace(0, 1.0, n_frames)

    n_tanh = 200
    W, b = generate_tanh_weights(n_tanh, 7, scale=3.5, ds=2.5)
    x_ic = np.linspace(-3.0, 3.0, 2000)
    y_ic = -np.sin(np.pi * x_ic)
    beta, bias = fit_elm(x_ic, y_ic, W, b, lam=1e-10)
    rmse = np.sqrt(np.mean((elm_eval(x_ic, W, b, beta, bias) - y_ic)**2))
    print(f"IC fit: {n_tanh} neurons, RMSE={rmse:.2e}")

    # ── Pre-compute all frames ──
    print("Pre-computing unified CIELM frames...")
    t0 = time.time()
    our_data = []
    for i, t in enumerate(t_values):
        u_exact = exact_colehopf(x_eval, t, nu=0.001)
        u_pred, shock_info = unified_cielm(x_eval, t, W, b, beta, bias)
        norm = max(np.linalg.norm(u_exact), 1e-12)
        l2 = float(np.linalg.norm(u_pred - u_exact) / norm)
        our_data.append({
            't': t, 'u_exact': u_exact.copy(), 'u_pred': u_pred.copy(),
            'l2': l2, 'shock': shock_info
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_frames}  t={t:.3f}  L2={l2:.2e}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Render comparison frames ──
    print("Rendering comparison frames...")
    combined = []

    for i in range(n_frames):
        cd = our_data[i]
        t = cd['t']

        fig = plt.figure(figsize=(16, 6.5), facecolor='white')

        # LEFT: Caio's PSO-PINN
        ax_left = fig.add_axes([0.02, 0.08, 0.46, 0.78])
        caio_gif.seek(i)
        frame_img = caio_gif.copy().convert('RGB')
        w_img, h_img = frame_img.size
        crop = (int(w_img * 0.06), int(h_img * 0.04),
                int(w_img * 0.96), int(h_img * 0.92))
        ax_left.imshow(frame_img.crop(crop))
        ax_left.axis('off')
        ax_left.set_title('PSO-PINN  (viscous $\\nu=0.01$)',
                          fontsize=12, color=C_TEXT, fontfamily='serif', pad=8)

        # RIGHT: Unified CIELM
        ax = fig.add_axes([0.55, 0.08, 0.42, 0.78])
        ax.set_facecolor('#fafafa')
        ax.grid(True, color='#e0e0e0', lw=0.6, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_SPINE)

        # Exact
        ax.plot(x_eval, cd['u_exact'], color=C_EXACT, lw=2.8, zorder=3,
                label='Exact (Cole-Hopf)', solid_capstyle='round')

        # Our prediction
        ax.plot(x_eval, cd['u_pred'], color=C_PRED, lw=2.0, ls='--', zorder=4,
                label='Unified CIELM', dash_capstyle='round')

        # Shock marker
        if cd['shock'] is not None:
            xs = cd['shock']['x_shock']
            ax.axvline(xs, color=C_SHOCK, lw=1.5, ls=':', alpha=0.6, zorder=2)
            ax.plot(xs, 0, 's', color=C_SHOCK, ms=7, zorder=6)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlabel('$x$', fontsize=12, fontfamily='serif')
        ax.set_ylabel('$u(x,t)$', fontsize=12, fontfamily='serif')
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9.5, framealpha=0.95, edgecolor=C_SPINE,
                  loc='upper right', prop={'family': 'serif'})

        ax.set_title('Unified CIELM  (inviscid $\\nu \\to 0$)',
                      fontsize=12, color=C_TEXT, fontfamily='serif', pad=8)

        # Phase label
        phase = 'PRE-SHOCK' if t < t_break else 'POST-SHOCK'
        tc = C_TEXT if t < t_break else C_SHOCK
        ax.text(0.03, 0.96, f'$t = {t:.4f}$   [{phase}]',
                transform=ax.transAxes, fontsize=10.5,
                va='top', ha='left', color=tc, fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_SPINE, alpha=0.9))

        # Info box
        if cd['shock'] is not None:
            info_str = (f"Shock @ $x={cd['shock']['x_shock']:+.3f}$\n"
                        f"Jump = {cd['shock']['jump']:.3f}\n"
                        f"L$_2$ = {cd['l2']:.2e}")
        else:
            info_str = f"Smooth (no shock)\nL$_2$ = {cd['l2']:.2e}"

        ax.text(0.03, 0.82, info_str,
                transform=ax.transAxes, fontsize=8.5,
                va='top', ha='left', color=C_TEXT, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_SPINE, alpha=0.85))

        # Supertitle
        fig.suptitle(
            r'Burgers Equation: $u_t + u\,u_x = \nu\,u_{xx}$,   '
            r'$u_0(x) = -\sin(\pi x)$',
            fontsize=14, fontfamily='serif', color=C_TEXT, y=0.96)

        # Render to PIL
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white')
        buf.seek(0)
        combined.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        gc.collect()

        if (i + 1) % 25 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    # ── Save GIF ──
    out_path = os.path.join(RESULTS_DIR, 'pso_vs_unified_cielm.gif')
    print(f"Saving: {out_path}")
    # 70ms per frame (~14 fps), hold last frame longer
    durations = [70] * (n_frames - 1) + [280]
    combined[0].save(out_path, save_all=True, append_images=combined[1:],
                     duration=durations, loop=0)
    fsize = os.path.getsize(out_path) / 1024 / 1024
    print(f"Done! {fsize:.1f} MB, {n_frames} frames, ~{n_frames*0.07:.1f}s")


if __name__ == '__main__':
    main()
