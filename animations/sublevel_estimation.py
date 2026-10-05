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
# LaTeX Typography & Styling (Computer Modern font matching thesis document)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


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

# Elliptic metric parameters (tilted ellipse)
ELLIPSE_THETA   = np.radians(-25)
COS_THETA       = float(np.cos(ELLIPSE_THETA))
SIN_THETA       = float(np.sin(ELLIPSE_THETA))
A_SCALE         = 1.35   # Major semi-axis scale
B_SCALE         = 0.75   # Minor semi-axis scale

def compute_v(x, y, k_quad=0.085):
    """Elliptic Lyapunov function with asymmetric boundary profile."""
    xi = COS_THETA * x + SIN_THETA * y
    eta = -SIN_THETA * x + COS_THETA * y
    quad = (xi / A_SCALE)**2 + (eta / B_SCALE)**2
    # Base elliptic basin + slight non-convexity
    v = k_quad * quad + 0.012 * quad**2 + 0.015 * np.sin(1.6 * x) * np.cos(1.6 * y)
    # Subtle dip defining global minimum on the left boundary
    dip = -0.035 * np.exp(-((x + b_domain)**2 + (y - 0.59)**2) / 0.50)
    return np.maximum(v + dip, 0.015)

def get_edge_minima(k_quad=0.085, n_pts=400):
    """Computes the local minimum on each of the 4 boundary edges."""
    grid = np.linspace(-b_domain, b_domain, n_pts)
    # Top edge: y = +b_domain
    v_top = compute_v(grid, b_domain, k_quad=k_quad)
    top_pt = (float(grid[np.argmin(v_top)]), b_domain, float(np.min(v_top)))
    # Bottom edge: y = -b_domain
    v_bot = compute_v(grid, -b_domain, k_quad=k_quad)
    bot_pt = (float(grid[np.argmin(v_bot)]), -b_domain, float(np.min(v_bot)))
    # Right edge: x = +b_domain
    v_right = compute_v(b_domain, grid, k_quad=k_quad)
    right_pt = (b_domain, float(grid[np.argmin(v_right)]), float(np.min(v_right)))
    # Left edge: x = -b_domain
    v_left = compute_v(-b_domain, grid, k_quad=k_quad)
    left_pt = (-b_domain, float(grid[np.argmin(v_left)]), float(np.min(v_left)))
    
    # Left edge is the absolute global minimum
    return {"top": top_pt, "bot": bot_pt, "right": right_pt, "left": left_pt}

# Slide background color matching the left PowerPoint card
SLIDE_BG        = "#F2F2F2"   # Left card background (RGB: 242, 242, 242)

# -----------------------------------------------------------------------------
# 3. Matplotlib Setup (Compact Square for Card, 544x544 divisible by 16)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.44, 5.44), dpi=100)
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
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(colors=COLOR_DARK, labelsize=12.5, width=1.1)

    # Axis labels at the tips
    ax.text(2.18, -0.02, r"$x_1$", fontsize=15, color=COLOR_DARK, va="top", ha="left")
    ax.text(-0.02, 2.18, r"$x_2$", fontsize=15, color=COLOR_DARK, va="bottom", ha="right")

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
    ax.scatter(0, 0, s=36, color=COLOR_DARK, zorder=6)
    ax.text(-0.09, -0.16, r"$x^\star$", fontsize=13.0, color=COLOR_DARK)

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
edge_mins = get_edge_minima(k_quad=k_quad)
top_target = (edge_mins["top"][0], edge_mins["top"][1])
bot_target = (edge_mins["bot"][0], edge_mins["bot"][1])
right_target = (edge_mins["right"][0], edge_mins["right"][1])
left_target = (edge_mins["left"][0], edge_mins["left"][1])
min_x, min_y, min_z = edge_mins["left"]

# 4 boundary seeds on each of the 4 edges (16 seeds total)
top_seeds_init = [(-1.65, b_domain), (-0.65, b_domain), (0.35, b_domain), (1.45, b_domain)]
bot_seeds_init = [(-1.45, -b_domain), (-0.35, -b_domain), (0.65, -b_domain), (1.65, -b_domain)]
right_seeds_init = [(b_domain, -1.55), (b_domain, -0.75), (b_domain, 0.25), (b_domain, 1.25)]
left_seeds_init = [(-b_domain, -1.25), (-b_domain, -0.25), (-b_domain, 0.75), (-b_domain, 1.55)]

# Static introduction: Basin with small initial level set
rho_initial = 0.05
for _ in range(20):
    render_frame(k_quad, rho_initial)

# -----------------------------------------------------------------------------
# PHASE 1: Boundary Seeds Initialization (Uniform B_d ~ U(d X_V))
# -----------------------------------------------------------------------------
all_inits = top_seeds_init + bot_seeds_init + right_seeds_init + left_seeds_init
for f in range(12):
    alpha = min(1.0, (f + 1) / 8.0)
    seeds = [(x, y, alpha, 55) for x, y in all_inits]
    render_frame(k_quad, rho_initial, seeds=seeds)

for _ in range(10):
    seeds = [(x, y, 1.0, 55) for x, y in all_inits]
    render_frame(k_quad, rho_initial, seeds=seeds)

# -----------------------------------------------------------------------------
# PHASE 2: 5 Discrete PGD Steps into the Local Minimum of each Edge
# -----------------------------------------------------------------------------
n_pgd_steps = 5
frames_per_step = 7

for s in range(n_pgd_steps):
    s_start = s / float(n_pgd_steps)
    s_end = (s + 1) / float(n_pgd_steps)
    
    for f in range(frames_per_step):
        # Progress within this step
        sub_frac = (f + 1) / float(frames_per_step)
        curr_frac = s_start + (s_end - s_start) * (sub_frac ** 1.2)
        
        current_seeds = []
        # Top edge seeds move towards top local minimum
        for x0, y0 in top_seeds_init:
            cx = (1 - curr_frac) * x0 + curr_frac * top_target[0]
            cy = (1 - curr_frac) * y0 + curr_frac * top_target[1]
            current_seeds.append((cx, cy, 1.0, 55))
        # Bottom edge seeds move towards bottom local minimum
        for x0, y0 in bot_seeds_init:
            cx = (1 - curr_frac) * x0 + curr_frac * bot_target[0]
            cy = (1 - curr_frac) * y0 + curr_frac * bot_target[1]
            current_seeds.append((cx, cy, 1.0, 55))
        # Right edge seeds move towards right local minimum
        for x0, y0 in right_seeds_init:
            cx = (1 - curr_frac) * x0 + curr_frac * right_target[0]
            cy = (1 - curr_frac) * y0 + curr_frac * right_target[1]
            current_seeds.append((cx, cy, 1.0, 55))
        # Left edge seeds move towards left local minimum (Global)
        for x0, y0 in left_seeds_init:
            cx = (1 - curr_frac) * x0 + curr_frac * left_target[0]
            cy = (1 - curr_frac) * y0 + curr_frac * left_target[1]
            current_seeds.append((cx, cy, 1.0, 55))
            
        render_frame(k_quad, rho_initial, seeds=current_seeds)

# Hold at the 4 local edge minima
edge_converged_seeds = [
    (top_target[0], top_target[1], 1.0, 65),
    (bot_target[0], bot_target[1], 1.0, 65),
    (right_target[0], right_target[1], 1.0, 65),
    (left_target[0], left_target[1], 1.0, 65),
]
for _ in range(12):
    render_frame(k_quad, rho_initial, seeds=edge_converged_seeds)

# -----------------------------------------------------------------------------
# PHASE 2b: Sub-optimal local minima disappear; ONLY absolute minimum remains
# -----------------------------------------------------------------------------
n_elim = 16
for f in range(n_elim):
    alpha_sub = max(0.0, 1.0 - (f + 1) / 10.0)
    alpha_global = min(1.0, (f + 1) / 8.0)
    
    # Sub-optimal minima (top, bot, right) fade out
    sub_seeds = [
        (top_target[0], top_target[1], alpha_sub, 65),
        (bot_target[0], bot_target[1], alpha_sub, 65),
        (right_target[0], right_target[1], alpha_sub, 65),
    ] if alpha_sub > 0 else None
    
    # Global minimum (left) transitions to Red
    min_pt = (min_x, min_y, alpha_global)
    render_frame(k_quad, rho_initial, seeds=sub_seeds, min_pt=min_pt)

# Hold on the unique global boundary minimum
for _ in range(12):
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

# Hold at the exact tangent contact point (visual proof of domain bound sublevel estimate)
for _ in range(50):
    render_frame(k_quad, min_z, min_pt=(min_x, min_y, 1.0))

writer.close()
plt.close(fig)
print("Animation completed successfully: sublevel_estimation.mp4")
