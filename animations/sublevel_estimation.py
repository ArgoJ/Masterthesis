"""
2D Sublevel Set Estimation Animation (Matplotlib LaTeX Style)
------------------------------------------------------------
Visualizes rho-sublevel estimation via boundary minimum search (Section 5.2.4):
- 2D state space (x1, x2) with training domain boundary d X_V (dashed box)
- Background iso-contours of Lyapunov function V(x)
- Uniform boundary seeds B_d ~ U(d X_V) appear along the perimeter (Teal)
- Projected Gradient Descent (PGD): Seeds slide along boundary edges into the
  global boundary minimum point rho_max = min_{x in d X_V} V(x) (Red)
- Sublevel set V_rho expands smoothly until touching the boundary at the exact
  location of that minimum point (tangent contact)
- Optimization / steepening step lifts boundary values, expanding V_rho further
- Minimalist: only axes (x1, x2) and origin (x*), no text overlays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

# -----------------------------------------------------------------------------
# 1. Thesis Color Palette (matching pgfplotssetup.tex & defence)
# -----------------------------------------------------------------------------
COLOR_NAVY      = "#1B365D"   # Primary thesis blue (domain boundary)
COLOR_SUBLEVEL  = "#D97706"   # Amber: Sublevel set V_rho
COLOR_MIN_PT    = "#DC2626"   # Vivid Red: Critical boundary minimum point rho = min V
COLOR_SEEDS     = "#0D9488"   # Teal: PGD boundary search seeds
COLOR_DARK      = "#1F2937"   # Structural: Axes, origin marker
COLOR_CONTOUR   = "#94A3B8"   # Slate gray: Iso-contours of V(x)

# -----------------------------------------------------------------------------
# 2. State Space & Lyapunov Function Mathematics
# -----------------------------------------------------------------------------
n_grid = 140
b_domain = 1.95  # Domain half-width: X_V = [-b, b] x [-b, b]
x_vals = np.linspace(-2.15, 2.15, n_grid)
y_vals = np.linspace(-2.15, 2.15, n_grid)
X, Y = np.meshgrid(x_vals, y_vals)

def compute_v(x, y, k_quad=0.085):
    """Lyapunov candidate with asymmetric non-convex boundary profile."""
    base = k_quad * (x**2 + y**2) + 0.012 * (x**4 + y**4) / 4.0
    asym = 0.035 * np.sin(1.8 * x) * np.cos(1.8 * y)
    # Localized dip near top boundary (x = 0.25, y = b_domain) defining global boundary minimum
    dip = -0.065 * np.exp(-((x - 0.25)**2 + (y - b_domain)**2) / 0.45)
    return np.maximum(base + asym + dip, 0.015)

# Boundary perimeter discretization (4 edges counter-clockwise)
n_edge = 100
edge_vals = np.linspace(-b_domain, b_domain, n_edge)

def get_boundary_points(k_quad=0.085):
    top_x, top_y = edge_vals, np.full(n_edge, b_domain)
    right_x, right_y = np.full(n_edge, b_domain), edge_vals[::-1]
    bot_x, bot_y = edge_vals[::-1], np.full(n_edge, -b_domain)
    left_x, left_y = np.full(n_edge, -b_domain), edge_vals
    
    bx = np.concatenate([top_x, right_x[1:], bot_x[1:], left_x[1:]])
    by = np.concatenate([top_y, right_y[1:], bot_y[1:], left_y[1:]])
    bz = compute_v(bx, by, k_quad=k_quad)
    return bx, by, bz

def get_boundary_min(k_quad=0.085):
    bx, by, bz = get_boundary_points(k_quad=k_quad)
    min_idx = np.argmin(bz)
    return float(bx[min_idx]), float(by[min_idx]), float(bz[min_idx]), min_idx

# Slide background color matching the left PowerPoint card
SLIDE_BG        = "#F2F2F2"   # Left card background (RGB: 242, 242, 242)

# -----------------------------------------------------------------------------
# 3. Matplotlib Setup (Compact Square for Card)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=100)
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.94)
output_filename = "sublevel_estimation.mp4"
writer = imageio.get_writer(output_filename, fps=30, quality=9)
print("Rendering 2D Sublevel Estimation Animation (MP4)...")

def render_frame(k_quad, rho_val, seeds=None, min_pt=None):
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
        linewidth=1.3, edgecolor=COLOR_NAVY, facecolor="none", linestyle="--", alpha=0.60
    )
    ax.add_patch(dom_box)

    # Compute current V field
    V_field = compute_v(X, Y, k_quad=k_quad)

    # 1. Subtle Iso-Contours of V(x) in the background
    contour_levels = np.linspace(0.04, float(np.max(V_field)) * 0.90, 12)
    ax.contour(
        X, Y, V_field,
        levels=contour_levels,
        colors=COLOR_CONTOUR,
        linewidths=0.65,
        alpha=0.45
    )

    # 2. Sublevel set V_rho = {x | V(x) <= rho} (Amber filled region & solid boundary)
    if rho_val > 0.01:
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

    # 4. PGD Boundary Seeds (Teal)
    if seeds is not None:
        for sx, sy, alpha_s, s_size in seeds:
            if alpha_s > 0:
                ax.scatter(sx, sy, s=s_size, color=COLOR_SEEDS, edgecolors="white", linewidth=1.0, alpha=alpha_s, zorder=7)

    # 5. Critical Boundary Minimum Point rho_max = min_{x in d X_V} V(x) (Red)
    if min_pt is not None:
        mx, my, alpha_m = min_pt
        ax.scatter(mx, my, s=110, color=COLOR_MIN_PT, edgecolors="white", linewidth=1.4, alpha=alpha_m, zorder=8)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    writer.append_data(rgba[:h - h%2, :w - w%2, :3])

# =============================================================================
# ANIMATION LOOP
# =============================================================================

k_quad = 0.085
bx_all, by_all, bz_all = get_boundary_points(k_quad=k_quad)
min_x, min_y, min_z, target_idx = get_boundary_min(k_quad=k_quad)

# 16 boundary seed positions distributed along perimeter
seed_indices = np.linspace(0, len(bx_all) - 1, 16, dtype=int, endpoint=False)

# Static introduction: Basin with small initial level set
rho_initial = 0.06
for _ in range(25):
    render_frame(k_quad, rho_initial)

# -----------------------------------------------------------------------------
# PHASE 1: Boundary Seeds Initialization (Uniform B_d ~ U(d X_V))
# -----------------------------------------------------------------------------
for f in range(15):
    alpha = min(1.0, (f + 1) / 10.0)
    seeds = [(bx_all[idx], by_all[idx], alpha, 55) for idx in seed_indices]
    render_frame(k_quad, rho_initial, seeds=seeds)

for _ in range(15):
    seeds = [(bx_all[idx], by_all[idx], 1.0, 55) for idx in seed_indices]
    render_frame(k_quad, rho_initial, seeds=seeds)

# -----------------------------------------------------------------------------
# PHASE 2: Projected Gradient Descent (PGD) along Boundary Edges
# -----------------------------------------------------------------------------
n_pgd = 45
for step in range(n_pgd):
    frac = (step + 1) / n_pgd
    seeds = []
    for start_idx in seed_indices:
        diff = (target_idx - start_idx)
        # Shortest distance around boundary loop
        diff = (diff + len(bx_all) // 2) % len(bx_all) - len(bx_all) // 2
        curr_idx = int((start_idx + diff * (frac**1.4)) % len(bx_all))
        seeds.append((bx_all[curr_idx], by_all[curr_idx], 1.0, 55))
        
    render_frame(k_quad, rho_initial, seeds=seeds)

# Global minimum point appears in Red, intermediate seeds fade out
for f in range(15):
    alpha_m = min(1.0, (f + 1) / 10.0)
    alpha_s = max(0.0, 1.0 - (f + 1) / 10.0)
    seeds = [(bx_all[target_idx], by_all[target_idx], alpha_s, 55)] if alpha_s > 0 else None
    render_frame(k_quad, rho_initial, seeds=seeds, min_pt=(min_x, min_y, alpha_m))

for _ in range(15):
    render_frame(k_quad, rho_initial, min_pt=(min_x, min_y, 1.0))

# -----------------------------------------------------------------------------
# PHASE 3: Sublevel Set Expansion up to rho = min_{x in d X_V} V(x)
# -----------------------------------------------------------------------------
rho_target = min_z
n_expand = 45
for step in range(n_expand):
    alpha = (step + 1) / n_expand
    curr_rho = rho_initial + (rho_target - rho_initial) * alpha
    render_frame(k_quad, curr_rho, min_pt=(min_x, min_y, 1.0))

# Hold at the exact tangent contact point (visual proof of domain bound)
for _ in range(35):
    render_frame(k_quad, min_z, min_pt=(min_x, min_y, 1.0))

# -----------------------------------------------------------------------------
# PHASE 4: Basin Steepening / Optimization -> Boundary Minimum Rises -> Growth
# -----------------------------------------------------------------------------
n_grow = 50
k_start, k_end = 0.085, 0.170
for step in range(n_grow):
    alpha = (step + 1) / n_grow
    k_curr = k_start + (k_end - k_start) * alpha
    
    # Updated boundary minimum under steeper basin
    mx, my, mz, _ = get_boundary_min(k_quad=k_curr)
    render_frame(k_curr, mz, min_pt=(mx, my, 1.0))

# Final hold
for _ in range(40):
    mx, my, mz, _ = get_boundary_min(k_quad=k_end)
    render_frame(k_end, mz, min_pt=(mx, my, 1.0))

writer.close()
plt.close(fig)
print("Animation completed successfully: sublevel_estimation.mp4")
