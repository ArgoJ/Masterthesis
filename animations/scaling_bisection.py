"""
Scaling and Bisection Algorithm Animation (Matplotlib LaTeX Style)
------------------------------------------------------------------
Visualizes the two-phase certified value search for rho (Algorithm 1):
- Phase 1: Scaling phase (exponential growth with factor q)
- Phase 2: Bisection phase (binary search narrowing [rho_lo, rho_up])

Colors matching thesis MLPs:
- Teal (#0D9488): Certified / Verifiziert (Psi(rho) = True)
- Orange (#D97706): Uncertified / Nicht verifiziert (Psi(rho) = False)

Axes:
- x-axis: Steps i
- y-axis: rho
Output: scaling_bisection.mp4 (960x544, 16:9, LaTeX Computer Modern)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

# -----------------------------------------------------------------------------
# LaTeX Typography & Styling (Computer Modern font matching thesis document)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

# Colors matching MLPs
COLOR_TEAL        = "#0D9488"   # Certified (Psi = True)
COLOR_ORANGE      = "#D97706"   # Uncertified / Nicht verifiziert (Psi = False)
COLOR_DARK        = "#1F2937"   # Structural / Axes
COLOR_CONTOUR     = "#94A3B8"   # Slate gray
SLIDE_BG          = "#FFFFFF"

# -----------------------------------------------------------------------------
# Algorithm 1: Scaling & Bisection Trajectory Data
# -----------------------------------------------------------------------------
rho_true_limit = 2.345  # True boundary threshold
rho_0 = 0.50
q = 1.55

steps_all = []
rhos_all = []
cert_all = []
phase_all = []  # "scale" or "bisect"
rho_lo_hist = []
rho_up_hist = []

# Phase 1: Scaling
rho = rho_0
rho_lo = rho_0
rho_up = None

i = 0
while True:
    steps_all.append(i)
    rhos_all.append(rho)
    c = (rho <= rho_true_limit)
    cert_all.append(c)
    phase_all.append("scale")
    
    if c:
        rho_lo = rho
        rho_lo_hist.append(rho_lo)
        rho_up_hist.append(np.nan)
        rho = rho * q
        i += 1
    else:
        rho_up = rho
        rho_lo_hist.append(rho_lo)
        rho_up_hist.append(rho_up)
        i += 1
        break

split_step = i - 0.5
n_scale_steps = i

# Phase 2: Bisection
n_bisect = 7
for _ in range(n_bisect):
    rho_mid = 0.5 * (rho_lo + rho_up)
    steps_all.append(i)
    rhos_all.append(rho_mid)
    c = (rho_mid <= rho_true_limit)
    cert_all.append(c)
    phase_all.append("bisect")
    
    if c:
        rho_lo = rho_mid
    else:
        rho_up = rho_mid
        
    rho_lo_hist.append(rho_lo)
    rho_up_hist.append(rho_up)
    i += 1

total_evals = len(steps_all)
steps_all = np.array(steps_all)
rhos_all = np.array(rhos_all)
cert_all = np.array(cert_all)
rho_lo_hist = np.array(rho_lo_hist)
rho_up_hist = np.array(rho_up_hist)

# -----------------------------------------------------------------------------
# Matplotlib Figure & Video Writer Setup
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.44), dpi=100)
fig.subplots_adjust(left=0.08, right=0.74, bottom=0.14, top=0.88)
fig.patch.set_facecolor(SLIDE_BG)

output_filename = "scaling_bisection.mp4"
writer = imageio.get_writer(output_filename, fps=30, quality=9)
print("Rendering Scaling & Bisection Animation (MP4)...")

def render_state(current_step, current_progress=1.0, is_converged=False):
    ax.clear()
    ax.set_facecolor(SLIDE_BG)
    
    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_DARK)
    ax.spines["bottom"].set_color(COLOR_DARK)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xlim(-0.6, steps_all[-1] + 2.1)
    ax.set_ylim(0.0, max(rhos_all) * 1.20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlabel(r"$\mathrm{Step}\ i$", fontsize=16, color=COLOR_DARK, labelpad=9)
    ax.set_ylabel(r"$\rho$", fontsize=18, color=COLOR_DARK, labelpad=9)
    
    # Subtle reference line for true threshold (unlabeled to keep rho* uniquely on rho_lo)
    ax.axhline(rho_true_limit, color="#CBD5E1", linestyle=":", linewidth=1.2, alpha=0.70, zorder=1)

    # Phase divider line and simple headers
    if current_step >= n_scale_steps - 1:
        ax.axvline(split_step, color="#CBD5E1", linestyle="--", linewidth=1.2, zorder=1)
        ax.text(split_step / 2.0, max(rhos_all) * 1.09, "Scaling", 
                fontsize=15.5, color=COLOR_DARK, ha="center", va="center")
        ax.text(split_step + (steps_all[-1] - split_step) / 2.0, max(rhos_all) * 1.09, "Bisection", 
                fontsize=15.5, color=COLOR_DARK, ha="center", va="center")
    else:
        ax.text(n_scale_steps / 2.0, max(rhos_all) * 1.09, "Scaling", 
                fontsize=15.5, color=COLOR_DARK, ha="center", va="center")

    # Bisection uncertainty ribbon [rho_lo, rho_up]
    if current_step >= n_scale_steps:
        vis_mask = (steps_all >= n_scale_steps) & (steps_all <= current_step)
        if np.any(vis_mask):
            vis_steps = steps_all[vis_mask]
            vis_lo = rho_lo_hist[vis_mask]
            vis_up = rho_up_hist[vis_mask]
            
            fill_x = np.concatenate([[split_step], vis_steps])
            fill_lo = np.concatenate([[rho_lo_hist[n_scale_steps - 1]], vis_lo])
            fill_up = np.concatenate([[rho_up_hist[n_scale_steps - 1]], vis_up])
            
            ax.fill_between(fill_x, fill_lo, fill_up, color=COLOR_ORANGE, alpha=0.10, zorder=1)
            ax.plot(fill_x, fill_lo, color=COLOR_TEAL, linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
            ax.plot(fill_x, fill_up, color=COLOR_ORANGE, linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)

            # Explicitly label the bounds rho_up and rho_lo at the bisection start
            ax.text(split_step + 0.16, fill_up[0] + 0.08, r"$\rho_{\mathrm{up}}$", 
                    fontsize=14.5, color=COLOR_ORANGE, va="bottom", ha="left")
            ax.text(split_step + 0.16, fill_lo[0] - 0.09, r"$\rho_{\mathrm{lo}}$", 
                    fontsize=14.5, color=COLOR_TEAL, va="top", ha="left")

    # Connector line between evaluated steps
    if current_step > 0:
        ax.plot(steps_all[:current_step + 1], rhos_all[:current_step + 1], color="#CBD5E1", linewidth=1.2, zorder=3)

    # Stem lines
    for idx in range(current_step + 1):
        s = steps_all[idx]
        r = rhos_all[idx]
        c = cert_all[idx]
        col = COLOR_TEAL if c else COLOR_ORANGE
        ax.vlines(s, 0, r, color=col, linestyle=":", linewidth=1.1, alpha=0.45, zorder=2)

    # Points already evaluated
    for idx in range(current_step):
        s = steps_all[idx]
        r = rhos_all[idx]
        c = cert_all[idx]
        col = COLOR_TEAL if c else COLOR_ORANGE
        bg = "#E6FFFA" if c else "#FFFBEB"
        ax.scatter(s, r, s=105, facecolor=bg, edgecolors=col, linewidth=2.0, zorder=5)

    # Current point with evaluation pulse / appearance
    if current_step < total_evals:
        s = steps_all[current_step]
        r = rhos_all[current_step]
        c = cert_all[current_step]
        col = COLOR_TEAL if c else COLOR_ORANGE
        bg = "#E6FFFA" if c else "#FFFBEB"
        
        # Halo pulse
        halo_size = 105 * (1.0 + 1.2 * (1.0 - current_progress))
        halo_alpha = 0.35 * current_progress
        ax.scatter(s, r, s=halo_size, facecolor=col, alpha=halo_alpha, edgecolors="none", zorder=4)
        ax.scatter(s, r, s=105, facecolor=bg, edgecolors=col, linewidth=2.0, zorder=5)

    # Final convergence: clearly mark rho* = rho_lo
    if is_converged:
        final_lo = rho_lo_hist[-1]
        ax.scatter(steps_all[-1], final_lo, s=240, facecolor=COLOR_TEAL, edgecolors="white", linewidth=1.6, marker="*", zorder=7)
        ax.text(steps_all[-1] + 0.25, final_lo, r"$\rho^\star = \rho_{\mathrm{lo}}$", 
                fontsize=15.5, color=COLOR_TEAL, va="center", ha="left", zorder=8)

    # Legend beside plot, without box frame, simple labels
    leg_cert = ax.scatter([], [], s=95, facecolor="#E6FFFA", edgecolors=COLOR_TEAL, linewidth=2.0, 
                          label="Zertifiziert")
    leg_uncert = ax.scatter([], [], s=95, facecolor="#FFFBEB", edgecolors=COLOR_ORANGE, linewidth=2.0, 
                            label="Unzertifiziert")
    ax.legend(handles=[leg_cert, leg_uncert], loc="center left", bbox_to_anchor=(1.02, 0.5), 
              frameon=False, fontsize=13.5, handletextpad=0.6, borderpad=0.2)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    if writer is not None:
        writer.append_data(rgba[:h - h%2, :w - w%2, :3])

# =============================================================================
# ANIMATION LOOP
# =============================================================================

# Initial pause
for _ in range(18):
    render_state(0, current_progress=0.1)

# Step-by-step evaluation
for s_idx in range(total_evals):
    # Arrival pulse
    n_frames_pulse = 10
    for f in range(n_frames_pulse):
        prog = (f + 1) / n_frames_pulse
        render_state(s_idx, current_progress=prog)
        
    # Hold state
    for _ in range(10):
        render_state(s_idx, current_progress=1.0)

# Final hold with convergence marker
for _ in range(45):
    render_state(total_evals - 1, current_progress=1.0, is_converged=True)

writer.close()
plt.savefig("/home/josua/programming_stuff/projects/ma-tex/animations/scaling_bisection.png", dpi=120)
plt.close(fig)
print("Animation completed successfully: scaling_bisection.mp4 and scaling_bisection.png")
