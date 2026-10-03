"""
2D CEGIS Concept Animation (Matplotlib LaTeX Style)
---------------------------------------------------
Demonstrates the 3-step CEGIS loop for Lyapunov function synthesis and 
Region of Attraction (ROA) sublevel set expansion in a 2D state space:

- Pure 2D phase portrait (x1, x2) in clean LaTeX / Thesis aesthetic
- Subtle iso-contour lines of the Lyapunov function V(x)
- Sublevel set V_rho = {x | V(x) <= rho} in warm Amber (#D97706)
- Counterexamples (cex) appear in vivid Red (#DC2626) at defect regions
- Dynamic test trajectories show violation / get trapped, then adapt to reach origin
- As the basin steepens, rho grows outward across 3 CEGIS iterations
- Final state: strictly decreasing Lyapunov function with converging trajectory family
- Minimalist: only axes (x1, x2) and origin (x*), no intrusive text overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

# -----------------------------------------------------------------------------
# 1. Thesis Color Palette (matching pgfplotssetup.tex & defence)
# -----------------------------------------------------------------------------
COLOR_NAVY      = "#1B365D"   # Primary thesis blue (domain boundary, trajectories)
COLOR_SUBLEVEL  = "#D97706"   # Amber: Sublevel set V_rho
COLOR_CEX       = "#DC2626"   # Vivid Red: Counterexamples x_adv
COLOR_DARK      = "#1F2937"   # Structural: Axes, origin marker
COLOR_TEAL      = "#0D9488"   # Secondary: Trajectory family
COLOR_CONTOUR   = "#94A3B8"   # Slate gray: Iso-contours of V(x)

# -----------------------------------------------------------------------------
# 2. State Space Grid & Lyapunov Function Mathematics
# -----------------------------------------------------------------------------
n_grid = 140
b_domain = 1.95
x_vals = np.linspace(-2.15, 2.15, n_grid)
y_vals = np.linspace(-2.15, 2.15, n_grid)
X, Y = np.meshgrid(x_vals, y_vals)

# Defect centers for 3 concentric zones / steps
ce_step1 = [
    (0.60, 0.40),
    (-0.55, 0.45),
    (0.35, -0.60),
    (-0.50, -0.40),
]

ce_step2 = [
    (1.05, 0.30),
    (-0.90, 0.70),
    (0.65, -1.00),
    (-1.00, -0.50),
    (0.15, 1.15),
]

ce_step3 = [
    (1.40, 0.50),
    (-1.25, 0.80),
    (1.00, -1.20),
    (-0.75, -1.40),
    (-1.45, -0.25),
]

def gaussian_dent(x, y, cx, cy, depth=0.22, width=0.09):
    return -depth * np.exp(-((x - cx)**2 + (y - cy)**2) / width)

def defect_field_1(x, y):
    d = np.zeros_like(x)
    for cx, cy in ce_step1:
        d += gaussian_dent(x, y, cx, cy, depth=0.18, width=0.08)
    return d

def defect_field_2(x, y):
    d = np.zeros_like(x)
    for cx, cy in ce_step2:
        d += gaussian_dent(x, y, cx, cy, depth=0.18, width=0.10)
    return d

def defect_field_3(x, y):
    d = np.zeros_like(x)
    for cx, cy in ce_step3:
        d += gaussian_dent(x, y, cx, cy, depth=0.16, width=0.11)
    return d

def ripple_field(x, y):
    return 0.025 * np.sin(2.5 * x) * np.cos(2.5 * y)

def compute_v(x, y, k_quad=0.075, w1=1.0, w2=1.0, w3=1.0, wr=1.0):
    base = k_quad * (x**2 + y**2) + 0.015 * (x**4 + y**4) / 4.0
    val = (base 
           + w1 * defect_field_1(x, y) 
           + w2 * defect_field_2(x, y) 
           + w3 * defect_field_3(x, y) 
           + wr * ripple_field(x, y))
    return np.maximum(val, 0.015)

def get_domain_min_boundary_val(k_quad, w1, w2, w3, wr, n_edge=80):
    edge = np.linspace(-b_domain, b_domain, n_edge)
    v_top = compute_v(edge, np.full_like(edge, b_domain), k_quad, w1, w2, w3, wr)
    v_bottom = compute_v(edge, np.full_like(edge, -b_domain), k_quad, w1, w2, w3, wr)
    v_right = compute_v(np.full_like(edge, b_domain), edge, k_quad, w1, w2, w3, wr)
    v_left = compute_v(np.full_like(edge, -b_domain), edge, k_quad, w1, w2, w3, wr)
    return float(np.min([np.min(v_top), np.min(v_bottom), np.min(v_right), np.min(v_left)]))

# -----------------------------------------------------------------------------
# 3. Dynamics & Trajectory Simulation
# -----------------------------------------------------------------------------
def simulate_trajectory(x0, y0, t_max=2.5, steps=90):
    dt = t_max / (steps - 1)
    xs = np.zeros(steps)
    ys = np.zeros(steps)
    xs[0], ys[0] = x0, y0
    for i in range(steps - 1):
        xc, yc = xs[i], ys[i]
        r2 = xc**2 + yc**2
        dx = -0.45 * xc + 1.15 * yc - 0.04 * xc * r2
        dy = -1.15 * xc - 0.45 * yc - 0.04 * yc * r2
        xs[i+1] = xc + dt * dx
        ys[i+1] = yc + dt * dy
    return xs, ys

# Trajectories for the 3 iterations
traj1_x, traj1_y = simulate_trajectory(1.10, 0.15, t_max=2.2, steps=80)
traj2_x, traj2_y = simulate_trajectory(-1.35, 0.75, t_max=2.2, steps=80)
traj3_x, traj3_y = simulate_trajectory(1.60, -0.50, t_max=2.4, steps=85)

# Final trajectory bundle
final_trajs_xy = []
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    r_start = 1.65
    tx, ty = simulate_trajectory(r_start * np.cos(angle), r_start * np.sin(angle), t_max=3.0, steps=85)
    final_trajs_xy.append((tx, ty))

final_colors = [COLOR_NAVY, COLOR_TEAL, "#0F766E", "#1E3A8A", COLOR_NAVY, COLOR_TEAL, "#0F766E", "#1E3A8A"]

# Slide background color: pure white
SLIDE_BG        = "#FFFFFF"   # White background

# -----------------------------------------------------------------------------
# 4. Matplotlib Setup (Compact Square for Card)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=100)
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.94)
output_filename = "cegis_concept_2d.mp4"
writer = imageio.get_writer(output_filename, fps=30, quality=9)
print("Rendering 2D CEGIS Concept Animation (MP4)...")

def render_frame(k_quad, w1, w2, w3, wr, active_cexs=[], traj_data=None, final_bundle_frac=None):
    ax.clear()
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect("equal")
    fig.patch.set_facecolor(SLIDE_BG)
    ax.set_facecolor(SLIDE_BG)

    # Clean axes styling (only x1, x2 and ticks, no bulky titles)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_color(COLOR_DARK)
    ax.spines["bottom"].set_color(COLOR_DARK)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    
    ax.set_xticks([-2, -1, 1, 2])
    ax.set_yticks([-2, -1, 1, 2])
    ax.tick_params(colors=COLOR_DARK, labelsize=8.5, width=1.0)

    # Axis labels at the tips
    ax.text(2.18, -0.02, r"$x_1$", fontsize=11, fontweight="bold", color=COLOR_DARK, va="top", ha="left")
    ax.text(-0.02, 2.18, r"$x_2$", fontsize=11, fontweight="bold", color=COLOR_DARK, va="bottom", ha="right")

    # Training domain boundary d X_V (dashed box)
    dom_box = patches.Rectangle(
        (-b_domain, -b_domain), 2 * b_domain, 2 * b_domain,
        linewidth=1.2, edgecolor=COLOR_NAVY, facecolor="none", linestyle="--", alpha=0.55
    )
    ax.add_patch(dom_box)

    # Compute current V field
    V_field = compute_v(X, Y, k_quad, w1, w2, w3, wr)
    rho_val = get_domain_min_boundary_val(k_quad, w1, w2, w3, wr)

    # 1. Subtle Iso-Contours of V(x) in the background
    contour_levels = np.linspace(0.02, float(np.max(V_field)) * 0.95, 12)
    ax.contour(
        X, Y, V_field,
        levels=contour_levels,
        colors=COLOR_CONTOUR,
        linewidths=0.65,
        alpha=0.45
    )

    # 2. Sublevel set V_rho = {x | V(x) <= rho} (Amber filled region & solid boundary)
    ax.contourf(
        X, Y, V_field,
        levels=[0.0, rho_val],
        colors=[COLOR_SUBLEVEL],
        alpha=0.28
    )
    ax.contour(
        X, Y, V_field,
        levels=[rho_val],
        colors=[COLOR_SUBLEVEL],
        linewidths=2.0
    )

    # 3. Origin marker x* = (0,0)
    ax.scatter(0, 0, s=28, color=COLOR_DARK, zorder=6)
    ax.text(-0.08, -0.15, r"$x^\star$", fontsize=8.5, color=COLOR_DARK, fontweight="bold")

    # 4. Active Counterexamples
    for cx, cy, alpha_cex, r_cex in active_cexs:
        if alpha_cex > 0:
            ax.scatter(cx, cy, s=r_cex, color=COLOR_CEX, edgecolors="white", linewidth=1.1, alpha=alpha_cex, zorder=8)

    # 5. Diagnostic Test Trajectory
    if traj_data is not None:
        tx, ty = traj_data
        ax.plot(tx, ty, color=COLOR_NAVY, linewidth=2.0, zorder=7)
        ax.scatter(tx[-1], ty[-1], s=22, color=COLOR_NAVY, zorder=7)

    # 6. Final Trajectory Bundle
    if final_bundle_frac is not None:
        for idx, (tx, ty) in enumerate(final_trajs_xy):
            curr_len = max(2, int(final_bundle_frac * len(tx)))
            col = final_colors[idx]
            ax.plot(tx[:curr_len], ty[:curr_len], color=col, linewidth=1.8, zorder=7)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    writer.append_data(rgba[:h - h%2, :w - w%2, :3])

# =============================================================================
# ANIMATION LOOP
# =============================================================================

# Initial state: Lyapunov basin with initial rho sublevel set
k_quad = 0.075
w1, w2, w3, wr = 1.0, 1.0, 1.0, 1.0

for _ in range(25):
    render_frame(k_quad, w1, w2, w3, wr)

# -----------------------------------------------------------------------------
# ITERATION 1: Counterexamples in inner zone -> Inductive synthesis -> Growth
# -----------------------------------------------------------------------------
# Trajectory 1 enters and gets trapped near dent 1
for f in range(24):
    idx = max(2, int((f + 1) / 24 * 38))
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj1_x[:idx], traj1_y[:idx]))

# 1. Counterexamples appear
for f in range(12):
    alpha = min(1.0, (f + 1) / 8.0)
    cexs = [(cx, cy, alpha, 70) for cx, cy in ce_step1]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj1_x[:38], traj1_y[:38]))

for _ in range(15):
    cexs = [(cx, cy, 1.0, 70) for cx, cy in ce_step1]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj1_x[:38], traj1_y[:38]))

# 2. Inductive synthesis: w1 -> 0 (dent disappears, trajectory flows to origin)
n_synth1 = 30
for f in range(n_synth1):
    prog = (f + 1) / n_synth1
    w1 = 1.0 - prog
    wr = 1.0 - 0.33 * prog
    
    alpha_ce = max(0.0, 1.0 - prog)
    cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step1]
    
    # Trajectory reaches origin
    idx = max(2, int(38 + prog * 42))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj1_x[:idx], traj1_y[:idx]))

# Hold solved trajectory at origin, then remove
for _ in range(10):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj1_x, traj1_y))

# Trajectory removed
for _ in range(6):
    render_frame(k_quad, w1, w2, w3, wr)

# 3. Learner steepens basin: k_quad grows from 0.075 to 0.155 (rho expands!)
n_grow1 = 40
k_start1, k_end1 = 0.075, 0.155
for f in range(n_grow1):
    k_quad = k_start1 + (k_end1 - k_start1) * (f + 1) / n_grow1
    render_frame(k_quad, w1, w2, w3, wr)

# -----------------------------------------------------------------------------
# ITERATION 2: New CEX in expanded sublevel set -> Resolve & Growth
# -----------------------------------------------------------------------------
# Trajectory 2 approaches middle zone defect
for f in range(20):
    idx = max(2, int((f + 1) / 20 * 40))
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj2_x[:idx], traj2_y[:idx]))

# 1. New counterexamples appear in expanded region
for f in range(12):
    alpha = min(1.0, (f + 1) / 8.0)
    cexs = [(cx, cy, alpha, 70) for cx, cy in ce_step2]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj2_x[:40], traj2_y[:40]))

for _ in range(15):
    cexs = [(cx, cy, 1.0, 70) for cx, cy in ce_step2]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj2_x[:40], traj2_y[:40]))

# 2. Inductive synthesis: w2 -> 0
n_synth2 = 30
for f in range(n_synth2):
    prog = (f + 1) / n_synth2
    w2 = 1.0 - prog
    wr = 0.67 - 0.33 * prog
    
    alpha_ce = max(0.0, 1.0 - prog)
    cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step2]
    
    idx = max(2, int(40 + prog * 40))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj2_x[:idx], traj2_y[:idx]))

for _ in range(10):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj2_x, traj2_y))

# Remove trajectory 2
for _ in range(6):
    render_frame(k_quad, w1, w2, w3, wr)

# 3. Learner steepens basin: k_quad grows from 0.155 to 0.280 (rho expands further!)
n_grow2 = 40
k_start2, k_end2 = 0.155, 0.280
for f in range(n_grow2):
    k_quad = k_start2 + (k_end2 - k_start2) * (f + 1) / n_grow2
    render_frame(k_quad, w1, w2, w3, wr)

# -----------------------------------------------------------------------------
# ITERATION 3: Outer zone defects -> Resolve -> Fully certified ROA
# -----------------------------------------------------------------------------
# Trajectory 3 approaches outer defect
for f in range(20):
    idx = max(2, int((f + 1) / 20 * 42))
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj3_x[:idx], traj3_y[:idx]))

# 1. New counterexamples appear in outer band
for f in range(12):
    alpha = min(1.0, (f + 1) / 8.0)
    cexs = [(cx, cy, alpha, 70) for cx, cy in ce_step3]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj3_x[:42], traj3_y[:42]))

for _ in range(15):
    cexs = [(cx, cy, 1.0, 70) for cx, cy in ce_step3]
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj3_x[:42], traj3_y[:42]))

# 2. Inductive synthesis: w3 -> 0 (surface becomes fully convex)
n_synth3 = 30
for f in range(n_synth3):
    prog = (f + 1) / n_synth3
    w3 = 1.0 - prog
    wr = max(0.0, 0.34 * (1.0 - prog))
    
    alpha_ce = max(0.0, 1.0 - prog)
    cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step3]
    
    idx = max(2, int(42 + prog * 43))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=(traj3_x[:idx], traj3_y[:idx]))

for _ in range(10):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj3_x, traj3_y))

# Remove trajectory 3
for _ in range(8):
    render_frame(k_quad, w1, w2, w3, wr)

# -----------------------------------------------------------------------------
# FINAL STATE: Strictly Decreasing Lyapunov Function & Converging Trajectory Family
# -----------------------------------------------------------------------------
n_final = 55
for f in range(n_final):
    frac = min(1.0, (f + 1) / 38.0)
    render_frame(k_quad, w1, w2, w3, wr, final_bundle_frac=frac)

for _ in range(35):
    render_frame(k_quad, w1, w2, w3, wr, final_bundle_frac=1.0)

writer.close()
plt.close(fig)
print("Animation completed successfully: cegis_concept_2d.mp4")
