"""
2D CEGIS Concept Animation with Dual Neural Networks & Lyapunov Condition (LaTeX Style)
----------------------------------------------------------------------------------------
Left:
- Policy Network pi_theta(x) completely in Teal (#0D9488): produces control u
- Lyapunov Network V_phi(x) completely in Amber (#D97706): produces energy V
- Between them: Discrete-time Lyapunov condition check:
    V_phi(x^+) > (1 - kappa) V_phi(x),  with x^+ = f(x, u)
- When a stability violation occurs, the condition block pulses Red and emits
  a Counterexample (CEX) particle that flies into the phase portrait to the defect!
- Forward pass pulses once through layers during trajectory rollout
- Backward pass runs once upon CEX detection, gently updating weights layer-by-layer
- Full reset after backprop: clean default state before the next phase

Right:
- 2D Phase Portrait (x1, x2) in authentic Computer Modern LaTeX aesthetic
- Lyapunov iso-contours & Sublevel set V_rho = {x | V(x) <= rho} (boundary minimum)
- Trajectory flows, gets trapped, receives CEX, and upon model update flows to origin
- Final state: certified Lyapunov function with converging trajectory family
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
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
# 1. Thesis Color Palette
# -----------------------------------------------------------------------------
COLOR_NAVY      = "#1B365D"   # Primary thesis blue (domain boundary, trajectories)
COLOR_SUBLEVEL  = "#D97706"   # Amber: Sublevel set V_rho & Lyapunov MLP
COLOR_CEX       = "#DC2626"   # Vivid Red: Counterexamples x_adv
COLOR_DARK      = "#1F2937"   # Structural: Axes, origin marker, text
COLOR_TEAL      = "#0D9488"   # Teal: Policy MLP
COLOR_CONTOUR   = "#94A3B8"   # Slate gray: Iso-contours of V(x)
SLIDE_BG        = "#FFFFFF"   # Pure white background

# -----------------------------------------------------------------------------
# 2. State Space Grid & Lyapunov Function Mathematics
# -----------------------------------------------------------------------------
n_grid = 140
b_domain = 1.95
x_vals = np.linspace(-2.15, 2.15, n_grid)
y_vals = np.linspace(-2.15, 2.15, n_grid)
X, Y = np.meshgrid(x_vals, y_vals)

ELLIPSE_THETA   = np.radians(-25)
COS_THETA       = float(np.cos(ELLIPSE_THETA))
SIN_THETA       = float(np.sin(ELLIPSE_THETA))
A_SCALE         = 1.35
B_SCALE         = 0.75

def ellipse_metric(x, y):
    xi = COS_THETA * x + SIN_THETA * y
    eta = -SIN_THETA * x + COS_THETA * y
    return (xi / A_SCALE)**2 + (eta / B_SCALE)**2

def get_ellipse_point(angle_rad, scale):
    xi = scale * A_SCALE * np.cos(angle_rad)
    eta = scale * B_SCALE * np.sin(angle_rad)
    x = COS_THETA * xi - SIN_THETA * eta
    y = SIN_THETA * xi + COS_THETA * eta
    return float(x), float(y)

# -----------------------------------------------------------------------------
# 3. Dynamics & Trajectory Simulation (Elliptic Spiral Inward Flow)
# -----------------------------------------------------------------------------
def simulate_trajectory(x0, y0, t_max=3.5, steps=90):
    dt = t_max / (steps - 1)
    xs, ys = np.zeros(steps), np.zeros(steps)
    xs[0], ys[0] = x0, y0
    for i in range(steps - 1):
        xc, yc = xs[i], ys[i]
        xi = COS_THETA * xc + SIN_THETA * yc
        eta = -SIN_THETA * xc + COS_THETA * yc
        d_xi = -0.75 * xi + 1.35 * (A_SCALE / B_SCALE) * eta
        d_eta = -1.35 * (B_SCALE / A_SCALE) * xi - 0.75 * eta
        dx = COS_THETA * d_xi - SIN_THETA * d_eta
        dy = SIN_THETA * d_xi + COS_THETA * d_eta
        xs[i+1] = xc + dt * dx
        ys[i+1] = yc + dt * dy
    return xs, ys

# Trajectories starting on boundary of estimated sublevel set (scale ~ 1.25)
x1_0, y1_0 = get_ellipse_point(np.radians(50), scale=1.25)
traj1_x, traj1_y = simulate_trajectory(x1_0, y1_0, t_max=3.5, steps=90)
p1_trap = (float(traj1_x[36]), float(traj1_y[36]))

x2_0, y2_0 = get_ellipse_point(np.radians(150), scale=1.25)
traj2_x, traj2_y = simulate_trajectory(x2_0, y2_0, t_max=3.5, steps=90)
p2_trap = (float(traj2_x[36]), float(traj2_y[36]))

x3_0, y3_0 = get_ellipse_point(np.radians(-50), scale=1.25)
traj3_x, traj3_y = simulate_trajectory(x3_0, y3_0, t_max=3.5, steps=90)
p3_trap = (float(traj3_x[36]), float(traj3_y[36]))

# Defect centers
ce_step1 = [p1_trap, (-0.45, 0.35), (0.42, 0.25)]
ce_step2 = [p2_trap, (-0.65, 0.45), (0.15, -0.75), (-0.35, -0.65)]
ce_step3 = [p3_trap, (0.95, -0.60), (-1.10, 0.65), (0.35, 1.05)]

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
    em = ellipse_metric(x, y)
    base = k_quad * em + 0.012 * (em**2)
    val = (base 
           + w1 * defect_field_1(x, y) 
           + w2 * defect_field_2(x, y) 
           + w3 * defect_field_3(x, y) 
           + wr * ripple_field(x, y))
    return np.maximum(val, 0.015)

def get_domain_min_boundary_val(k_quad, w1=0.0, w2=0.0, w3=0.0, wr=0.0, n_edge=80):
    edge = np.linspace(-b_domain, b_domain, n_edge)
    v_top = compute_v(edge, np.full_like(edge, b_domain), k_quad, w1, w2, w3, wr)
    v_bottom = compute_v(edge, np.full_like(edge, -b_domain), k_quad, w1, w2, w3, wr)
    v_right = compute_v(np.full_like(edge, b_domain), edge, k_quad, w1, w2, w3, wr)
    v_left = compute_v(np.full_like(edge, -b_domain), edge, k_quad, w1, w2, w3, wr)
    return float(np.min([np.min(v_top), np.min(v_bottom), np.min(v_right), np.min(v_left)]))

# Final trajectory bundle
final_trajs_xy = []
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    x0, y0 = get_ellipse_point(angle, scale=1.25)
    tx, ty = simulate_trajectory(x0, y0, t_max=3.5, steps=90)
    final_trajs_xy.append((tx, ty))

final_colors = [COLOR_NAVY, COLOR_TEAL, "#0F766E", "#1E3A8A", COLOR_NAVY, COLOR_TEAL, "#0F766E", "#1E3A8A"]

# -----------------------------------------------------------------------------
# 4. Neural Network Geometry & Precomputed Weights
# -----------------------------------------------------------------------------
layer_counts = [2, 4, 3, 1]
# Compact layer coordinates shifted left to provide ample space for the condition pill
xs = [0.07, 0.20, 0.33, 0.46]

def get_node_coords(y_center):
    node_coords = []
    for l_idx, count in enumerate(layer_counts):
        lx = xs[l_idx]
        spacing = 0.068 if count == 4 else (0.086 if count == 3 else 0.108)
        start_y = y_center - (count - 1) * spacing / 2.0
        coords = [(lx, start_y + i * spacing) for i in range(count)]
        node_coords.append(coords)
    return node_coords

nodes_pi = get_node_coords(y_center=0.74)
nodes_v  = get_node_coords(y_center=0.26)

# Condition check block coordinates (wide pill)
chk_x = 0.73
chk_y = 0.50
chk_w = 0.48
chk_h = 0.14

# Weight sequences for 4 iteration stages
weight_states_pi = []
weight_states_v  = []
for it in range(4):
    rng_pi = np.random.RandomState(101 + it * 37)
    w_pi = [rng_pi.uniform(0.85, 1.45, size=(len(nodes_pi[g]), len(nodes_pi[g+1]))) for g in range(3)]
    weight_states_pi.append(w_pi)
    
    rng_v = np.random.RandomState(202 + it * 43)
    w_v = [rng_v.uniform(0.85, 1.45, size=(len(nodes_v[g]), len(nodes_v[g+1]))) for g in range(3)]
    weight_states_v.append(w_v)

def draw_network(ax, node_coords, title, out_name, accent_color, node_fill, 
                 fwd_layer=None, fwd_prog=0.0, bp_layer=None, bp_prog=0.0, 
                 iteration=0, all_weights=None):
    y_title = node_coords[1][-1][1] + 0.055
    ax.text(0.26, y_title, title, fontsize=15.0, color=accent_color, va="center", ha="center")
    
    curr_it = min(iteration, 3)
    next_it = min(iteration + 1, 3)
    w_curr_mats = all_weights[curr_it]
    w_next_mats = all_weights[next_it]
    
    for g in range(3):
        is_active_bp  = (bp_layer == g)
        has_updated   = (bp_layer is not None and g > bp_layer)
        is_active_fwd = (fwd_layer == g)
        
        W_c = w_curr_mats[g]
        W_n = w_next_mats[g]
        
        for i, (x1, y1) in enumerate(node_coords[g]):
            for j, (x2, y2) in enumerate(node_coords[g+1]):
                wc = W_c[i, j]
                wn = W_n[i, j]
                
                if is_active_bp:
                    if bp_prog < 0.45:
                        s = np.sin(bp_prog / 0.45 * np.pi / 2)
                        lw = wc * 1.20 * (1.0 + 0.35 * s)
                        alp = 0.95
                    else:
                        d = np.cos((bp_prog - 0.45) / 0.55 * np.pi / 2)
                        lw = (1.0 - d) * (wn * 1.25) + d * (wc * 1.20 * 1.35)
                        alp = 0.88
                    ax.plot([x1, x2], [y1, y2], color=accent_color, linewidth=lw, alpha=alp, zorder=3)
                elif is_active_fwd:
                    s = np.sin(fwd_prog * np.pi)
                    lw = wc * 1.20 * (1.0 + 0.32 * s)
                    ax.plot([x1, x2], [y1, y2], color=accent_color, linewidth=lw, alpha=0.92, zorder=3)
                elif has_updated:
                    ax.plot([x1, x2], [y1, y2], color=accent_color, linewidth=wn * 1.25, alpha=0.75, zorder=2)
                else:
                    ax.plot([x1, x2], [y1, y2], color=accent_color, linewidth=wc * 1.15, alpha=0.48, zorder=1)
                    
    # Draw nodes
    for l_idx, coords in enumerate(node_coords):
        is_layer_active = False
        if bp_layer is not None and bp_layer >= 0 and (l_idx == bp_layer or l_idx == bp_layer + 1):
            is_layer_active = True
        elif fwd_layer is not None and (l_idx == fwd_layer or l_idx == fwd_layer + 1 or (fwd_layer == 3 and l_idx == 3)):
            is_layer_active = True
        
        for n_idx, (nx, ny) in enumerate(coords):
            if is_layer_active:
                ax.scatter(nx, ny, s=260, facecolor=accent_color, alpha=0.25, edgecolors="none", zorder=4)
                face = accent_color
                edge = accent_color
            else:
                face = node_fill
                edge = accent_color
                
            ax.scatter(nx, ny, s=145, facecolor=face, edgecolors=edge, linewidth=1.5, zorder=5)
            
            if l_idx == 0:
                inp_lbl = r"$x_1$" if n_idx == 1 else r"$x_2$"
                ax.text(nx - 0.040, ny, inp_lbl, fontsize=12.5, color=COLOR_DARK, va="center", ha="right", zorder=6)
            elif l_idx == len(node_coords) - 1:
                ax.text(nx + 0.040, ny, out_name, fontsize=13.5, color=COLOR_DARK, va="center", ha="left", zorder=6)

# -----------------------------------------------------------------------------
# 5. Matplotlib Setup (16:9 Aspect Ratio: 960x544 divisible by 16)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(9.6, 5.44), dpi=100)
gs = fig.add_gridspec(1, 2, width_ratios=[0.90, 1.10], left=0.03, right=0.97, bottom=0.06, top=0.94, wspace=0.08)

ax_nn = fig.add_subplot(gs[0, 0])
ax_plot = fig.add_subplot(gs[0, 1])

output_filename = "cegis_concept_2d.mp4"
writer = imageio.get_writer(output_filename, fps=30, quality=9)
print("Rendering 2D CEGIS Concept with Dual Neural Networks & Lyapunov Condition (MP4)...")

def render_frame(k_quad, w1, w2, w3, wr, active_cexs=[], traj_data=None, final_bundle_frac=None, 
                 fwd_layer=None, fwd_prog=0.0, bp_layer=None, bp_prog=0.0, iteration=0,
                 chk_active=False, flying_cexs=None):
    fig.patch.set_facecolor(SLIDE_BG)
    
    # -------------------------------------------------------------------------
    # Draw Left Subplot: Neural Networks & Lyapunov Condition
    # -------------------------------------------------------------------------
    ax_nn.clear()
    ax_nn.set_xlim(0, 1)
    ax_nn.set_ylim(0, 1)
    ax_nn.axis("off")
    ax_nn.set_facecolor(SLIDE_BG)
    
    draw_network(ax_nn, nodes_pi, title=r"$\pi_\theta(x)$", out_name=r"$u$", 
                 accent_color=COLOR_TEAL, node_fill="#E6FFFA",
                 fwd_layer=fwd_layer, fwd_prog=fwd_prog,
                 bp_layer=bp_layer, bp_prog=bp_prog, 
                 iteration=iteration, all_weights=weight_states_pi)
    
    draw_network(ax_nn, nodes_v,  title=r"$V_\phi(x)$",  out_name=r"$V$", 
                 accent_color=COLOR_SUBLEVEL, node_fill="#FFFBEB",
                 fwd_layer=fwd_layer, fwd_prog=fwd_prog,
                 bp_layer=bp_layer, bp_prog=bp_prog, 
                 iteration=iteration, all_weights=weight_states_v)

    # Connecting arrows from u and V into the Lyapunov Condition block
    # Colors stay authentic Teal and Amber (never turn red)
    arr_u_col = COLOR_TEAL
    arr_v_col = COLOR_SUBLEVEL

    # Dynamic line width: pulse thicker during forward pass (when output/condition is reached)
    # and during backward pass (when backprop starts at the condition block)
    is_arrow_active = False
    arr_lw = 1.6
    arr_alpha = 0.80
    if fwd_layer == 3:
        is_arrow_active = True
        s = np.sin(fwd_prog * np.pi)
        arr_lw = 1.6 + 2.2 * s
        arr_alpha = 1.0
    elif bp_layer == 3:  # Initial backprop phase from condition block into MLP outputs
        is_arrow_active = True
        s = np.sin(bp_prog * np.pi)
        arr_lw = 1.6 + 2.2 * s
        arr_alpha = 1.0
    elif bp_layer is not None and bp_layer >= 0:
        # Subtle emphasis while backprop travels through the network
        arr_alpha = 0.90

    outtxt_w = 0.07
    eps = 0.02

    # Upper Arrow (from pi / u to Condition Block):
    ax_nn.annotate("", xy=(chk_x, chk_y + chk_h/2 + eps), xytext=(xs[-1] + outtxt_w + eps, 0.74),
                   arrowprops=dict(arrowstyle="->", color=arr_u_col, lw=arr_lw, alpha=arr_alpha,
                                   connectionstyle="angle,angleA=0,angleB=90,rad=45"))
    # Lower Arrow (from V to Condition Block):
    ax_nn.annotate("", xy=(chk_x, chk_y - chk_h/2 - eps), xytext=(xs[-1] + outtxt_w + eps, 0.26),
                   arrowprops=dict(arrowstyle="->", color=arr_v_col, lw=arr_lw, alpha=arr_alpha,
                                   connectionstyle="angle,angleA=0,angleB=90,rad=45"))

    # Lyapunov Condition Pill: V_phi(x^+) <= (1 - kappa) V_phi(x)
    box_bg = "#FEF2F2" if chk_active else "#F8FAFC"
    box_edge = COLOR_CEX if chk_active else "#CBD5E1"
    box_lw = 1.6 if chk_active else 1.1

    pill = patches.FancyBboxPatch(
        (chk_x - chk_w/2, chk_y - chk_h/2), chk_w, chk_h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor=box_bg, edgecolor=box_edge, linewidth=box_lw, zorder=6
    )
    ax_nn.add_patch(pill)

    if chk_active:
        # Decrease
        t_left  = TextArea(r"$V_\phi(x^+)$", textprops=dict(color=COLOR_DARK, size=12.5))
        t_mid   = TextArea(r"$\mathbf{\nless}$",   textprops=dict(color=COLOR_CEX, size=13.5))
        t_right = TextArea(r"$(1-\kappa)V_\phi(x)$", textprops=dict(color=COLOR_DARK, size=12.5))

        decrease_box = HPacker(children=[t_left, t_mid, t_right], align="center", pad=0, sep=1)
        ab = AnnotationBbox(decrease_box, (chk_x, chk_y + 0.026), frameon=False, box_alignment=(0.5, 0.5), zorder=7)
        ax_nn.add_artist(ab)

        # Invariance
        t_left  = TextArea(r"$f(\mathcal{S})$", textprops=dict(color=COLOR_DARK, size=12.5))
        t_mid   = TextArea(r"$\not\subseteq$",   textprops=dict(color=COLOR_CEX, size=13.5))
        t_right = TextArea(r"$\mathcal{S}$", textprops=dict(color=COLOR_DARK, size=12.5))

        invariance_box = HPacker(children=[t_left, t_mid, t_right], align="center", pad=0, sep=1)
        ab = AnnotationBbox(invariance_box, (chk_x, chk_y - 0.026), frameon=False, box_alignment=(0.5, 0.5), zorder=7)
        ax_nn.add_artist(ab)
    else:
        # Clean default: satisfied Lyapunov decrease condition
        ax_nn.text(chk_x, chk_y + 0.026, r"$V_\phi(x^+) < (1-\kappa)V_\phi(x)$", 
                   fontsize=12.0, color=COLOR_DARK, va="center", ha="center", zorder=7)
        ax_nn.text(chk_x, chk_y - 0.026, r"$f(\mathcal{S}) \subseteq \mathcal{S}$", 
                   fontsize=11.0, color="#64748B", va="center", ha="center", zorder=7)

    # -------------------------------------------------------------------------
    # Draw Right Subplot: Phase Portrait
    # -------------------------------------------------------------------------
    ax_plot.clear()
    ax_plot.set_xlim(-2.25, 2.25)
    ax_plot.set_ylim(-2.25, 2.25)
    ax_plot.set_aspect("equal")
    ax_plot.set_facecolor(SLIDE_BG)

    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_position("zero")
    ax_plot.spines["bottom"].set_position("zero")
    ax_plot.spines["left"].set_color(COLOR_DARK)
    ax_plot.spines["bottom"].set_color(COLOR_DARK)
    ax_plot.spines["left"].set_linewidth(1.2)
    ax_plot.spines["bottom"].set_linewidth(1.2)
    
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    ax_plot.tick_params(colors=COLOR_DARK, labelsize=12.0, width=1.1)

    ax_plot.text(2.18, -0.02, r"$x_1$", fontsize=15.0, color=COLOR_DARK, va="top", ha="left")
    ax_plot.text(-0.02, 2.18, r"$x_2$", fontsize=15.0, color=COLOR_DARK, va="bottom", ha="right")

    # Training domain boundary
    dom_box = patches.Rectangle(
        (-b_domain, -b_domain), 2 * b_domain, 2 * b_domain,
        linewidth=1.2, edgecolor=COLOR_NAVY, facecolor="none", linestyle="--", alpha=0.55
    )
    ax_plot.add_patch(dom_box)

    # Compute current V field and exact boundary minimum rho
    V_field = compute_v(X, Y, k_quad, w1, w2, w3, wr)
    rho_val = get_domain_min_boundary_val(k_quad, w1, w2, w3, wr)

    # 1. Iso-Contours of V(x)
    contour_levels = np.linspace(0.02, float(np.max(V_field)) * 0.95, 12)
    ax_plot.contour(
        X, Y, V_field,
        levels=contour_levels,
        colors=COLOR_CONTOUR,
        linewidths=0.65,
        alpha=0.45
    )

    # 2. Sublevel set V_rho
    ax_plot.contourf(
        X, Y, V_field,
        levels=[0.0, rho_val],
        colors=[COLOR_SUBLEVEL],
        alpha=0.28
    )
    ax_plot.contour(
        X, Y, V_field,
        levels=[rho_val],
        colors=[COLOR_SUBLEVEL],
        linewidths=2.0
    )

    # 3. Origin marker
    ax_plot.scatter(0, 0, s=36, color=COLOR_DARK, zorder=6)
    ax_plot.text(-0.09, -0.16, r"$x^\star$", fontsize=13.0, color=COLOR_DARK)

    # 4. Active Counterexamples
    for cx, cy, alpha_cex, r_cex in active_cexs:
        if alpha_cex > 0:
            ax_plot.scatter(cx, cy, s=r_cex, color=COLOR_CEX, edgecolors="white", linewidth=1.1, alpha=alpha_cex, zorder=8)

    # 5. Flying CEX particles from condition block to defects
    if flying_cexs:
        for fx, fy in flying_cexs:
            ax_plot.scatter(fx, fy, s=140, facecolor=COLOR_CEX, edgecolors="none", alpha=0.30, clip_on=False, zorder=11)
            ax_plot.scatter(fx, fy, s=75, facecolor=COLOR_CEX, edgecolors="white", linewidth=1.1, clip_on=False, zorder=12)

    # 6. Diagnostic Test Trajectory
    if traj_data is not None:
        tx, ty = traj_data
        ax_plot.plot(tx, ty, color=COLOR_NAVY, linewidth=2.0, zorder=7)
        ax_plot.scatter(tx[-1], ty[-1], s=22, color=COLOR_NAVY, zorder=7)

    # 7. Final Trajectory Bundle
    if final_bundle_frac is not None:
        for idx, (tx, ty) in enumerate(final_trajs_xy):
            curr_len = max(2, int(final_bundle_frac * len(tx)))
            col = final_colors[idx]
            ax_plot.plot(tx[:curr_len], ty[:curr_len], color=col, linewidth=1.8, zorder=7)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    writer.append_data(rgba[:h - h%2, :w - w%2, :3])

# =============================================================================
# ANIMATION LOOP HELPERS
# =============================================================================

def get_flight_source():
    inv_plot = ax_plot.transData.inverted()
    p_src_disp = ax_nn.transData.transform((chk_x + chk_w/2, chk_y))
    return inv_plot.transform(p_src_disp)

def run_single_forward_pass_rollout(traj_x, traj_y, k_quad, w1, w2, w3, wr, iteration):
    """Executes a strictly monotonic trajectory rollout to index 36 with a single forward pass."""
    total_rollout_steps = 36
    n_rollout = 24  # Total frames to reach defect smoothly
    fwd_duration = 15  # 5 frames for layer 0, 5 for layer 1, 5 for layer 2
    
    for f in range(n_rollout):
        idx = max(2, int((f + 1) / n_rollout * total_rollout_steps))
        
        if f < fwd_duration:
            f_layer = min(2, f // 5)
            f_prog = (f % 5 + 1) / 5.0
        elif f < fwd_duration + 5:
            f_layer = 3
            f_prog = (f - fwd_duration + 1) / 5.0
        else:
            f_layer = None
            f_prog = 0.0
            
        render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj_x[:idx], traj_y[:idx]), 
                     fwd_layer=f_layer, fwd_prog=f_prog, bp_layer=None, iteration=iteration)

def run_cex_violation_and_flight(ce_group, k_quad, w1, w2, w3, wr, traj_trap, iteration):
    """
    1. Condition block flashes Red as violation occurs.
    2. ALL counterexample particles emerge from the condition block and fly across to their respective points!
    3. CEX dots land and establish at their defect coordinates.
    """
    p_src = get_flight_source()
    
    # Delay: trajectory settles at defect
    for _ in range(8):
        render_frame(k_quad, w1, w2, w3, wr, traj_data=traj_trap, 
                     chk_active=False, bp_layer=None, iteration=iteration)
        
    # Violation triggers in the condition block
    for _ in range(6):
        render_frame(k_quad, w1, w2, w3, wr, traj_data=traj_trap, 
                     chk_active=True, bp_layer=None, iteration=iteration)
        
    # All CEX particles fly simultaneously from the decrease field to their points (slower, 32 frames)
    n_flight = 32
    for f in range(n_flight):
        t = (f + 1) / n_flight
        t_ease = 0.5 - 0.5 * np.cos(t * np.pi)
        current_particles = []
        for idx, (cx, cy) in enumerate(ce_group):
            arc_val = 0.35 * (1.0 if cy >= p_src[1] else -1.0) * (0.8 + 0.3 * (idx % 2))
            curr_x = (1 - t_ease) * p_src[0] + t_ease * cx
            curr_y = (1 - t_ease) * p_src[1] + t_ease * cy + arc_val * np.sin(t_ease * np.pi)
            current_particles.append((curr_x, curr_y))
            
        render_frame(k_quad, w1, w2, w3, wr, traj_data=traj_trap, 
                     chk_active=True, flying_cexs=current_particles, bp_layer=None, iteration=iteration)
        
    # Landed: all counterexamples are active at their target spots
    cexs = [(cx, cy, 1.0, 70) for cx, cy in ce_group]
    for _ in range(10):
        render_frame(k_quad, w1, w2, w3, wr, active_cexs=cexs, traj_data=traj_trap, 
                     chk_active=True, bp_layer=None, iteration=iteration)
    return cexs

def run_backward_pass_with_delays(k_quad, w1, w2, w3, wr, active_cexs, traj_trap, iteration):
    """
    Runs backward pass from condition block backwards through layers,
    then cleanly RESETS all layers before the Lyapunov function expands.
    """
    n_bp_frames = 5
    for layer in [3, 2, 1, 0]:
        for f in range(n_bp_frames):
            p = (f + 1) / n_bp_frames
            render_frame(k_quad, w1, w2, w3, wr, active_cexs=active_cexs, traj_data=traj_trap, 
                         chk_active=True, bp_layer=layer, bp_prog=p, iteration=iteration)
            
    # RESET IMMEDIATELY! Condition block and all layers return to clean default
    new_iteration = min(iteration + 1, 3)
    for _ in range(8):
        render_frame(k_quad, w1, w2, w3, wr, active_cexs=active_cexs, traj_data=traj_trap, 
                     chk_active=False, bp_layer=None, fwd_layer=None, iteration=new_iteration)
    return new_iteration

# =============================================================================
# ANIMATION LOOP
# =============================================================================

k_quad = 0.075
w1, w2, w3, wr = 1.0, 1.0, 1.0, 1.0

# Initial hold
for _ in range(15):
    render_frame(k_quad, w1, w2, w3, wr, bp_layer=None, iteration=0)

# -----------------------------------------------------------------------------
# ITERATION 1
# -----------------------------------------------------------------------------
# 1. Single forward pass as trajectory 1 rolls out
run_single_forward_pass_rollout(traj1_x, traj1_y, k_quad, w1, w2, w3, wr, iteration=0)

# 2. Condition block triggers and emits flying CEX particles into defect 1
cexs1 = run_cex_violation_and_flight(ce_step1, k_quad, w1, w2, w3, wr, (traj1_x[:36], traj1_y[:36]), iteration=0)

# 3. Backward pass runs, updates weights, and resets cleanly
it1 = run_backward_pass_with_delays(k_quad, w1, w2, w3, wr, cexs1, (traj1_x[:36], traj1_y[:36]), iteration=0)

# 4. Exactly as weights update: Lyapunov function enlarges & defect vanishes!
n_synth1 = 34
k_start1, k_end1 = 0.075, 0.155
for f in range(n_synth1):
    prog = (f + 1) / n_synth1
    w1 = 1.0 - prog
    wr = 1.0 - 0.33 * prog
    k_quad = k_start1 + (k_end1 - k_start1) * prog
    
    alpha_ce = max(0.0, 1.0 - prog)
    curr_cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step1]
    
    idx = max(2, int(36 + prog * (90 - 36)))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=curr_cexs, traj_data=(traj1_x[:idx], traj1_y[:idx]), 
                 chk_active=False, bp_layer=None, iteration=it1)

for _ in range(8):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj1_x, traj1_y), bp_layer=None, iteration=it1)

for _ in range(6):
    render_frame(k_quad, w1, w2, w3, wr, bp_layer=None, iteration=it1)

# -----------------------------------------------------------------------------
# ITERATION 2
# -----------------------------------------------------------------------------
# 1. Single forward pass as trajectory 2 rolls out
run_single_forward_pass_rollout(traj2_x, traj2_y, k_quad, w1, w2, w3, wr, iteration=it1)

# 2. Condition block triggers and emits flying CEX particles into defect 2
cexs2 = run_cex_violation_and_flight(ce_step2, k_quad, w1, w2, w3, wr, (traj2_x[:36], traj2_y[:36]), iteration=it1)

# 3. Backward pass runs and resets cleanly
it2 = run_backward_pass_with_delays(k_quad, w1, w2, w3, wr, cexs2, (traj2_x[:36], traj2_y[:36]), iteration=it1)

# 4. Basin steepens & defect vanishes, trajectory completes to origin
n_synth2 = 34
k_start2, k_end2 = 0.155, 0.280
for f in range(n_synth2):
    prog = (f + 1) / n_synth2
    w2 = 1.0 - prog
    wr = 0.67 - 0.33 * prog
    k_quad = k_start2 + (k_end2 - k_start2) * prog
    
    alpha_ce = max(0.0, 1.0 - prog)
    curr_cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step2]
    
    idx = max(2, int(36 + prog * (90 - 36)))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=curr_cexs, traj_data=(traj2_x[:idx], traj2_y[:idx]), 
                 chk_active=False, bp_layer=None, iteration=it2)

for _ in range(8):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj2_x, traj2_y), bp_layer=None, iteration=it2)

for _ in range(6):
    render_frame(k_quad, w1, w2, w3, wr, bp_layer=None, iteration=it2)

# -----------------------------------------------------------------------------
# ITERATION 3
# -----------------------------------------------------------------------------
# 1. Single forward pass as trajectory 3 rolls out
run_single_forward_pass_rollout(traj3_x, traj3_y, k_quad, w1, w2, w3, wr, iteration=it2)

# 2. Condition block triggers and emits flying CEX particles into defect 3
cexs3 = run_cex_violation_and_flight(ce_step3, k_quad, w1, w2, w3, wr, (traj3_x[:36], traj3_y[:36]), iteration=it2)

# 3. Backward pass runs and resets cleanly
it3 = run_backward_pass_with_delays(k_quad, w1, w2, w3, wr, cexs3, (traj3_x[:36], traj3_y[:36]), iteration=it2)

# 4. Surface fully convex, final trajectory reaches origin
n_synth3 = 34
for f in range(n_synth3):
    prog = (f + 1) / n_synth3
    w3 = 1.0 - prog
    wr = max(0.0, 0.34 * (1.0 - prog))
    
    alpha_ce = max(0.0, 1.0 - prog)
    curr_cexs = [(cx, cy, alpha_ce, 70 * alpha_ce) for cx, cy in ce_step3]
    
    idx = max(2, int(36 + prog * (90 - 36)))
    render_frame(k_quad, w1, w2, w3, wr, active_cexs=curr_cexs, traj_data=(traj3_x[:idx], traj3_y[:idx]), 
                 chk_active=False, bp_layer=None, iteration=it3)

for _ in range(8):
    render_frame(k_quad, w1, w2, w3, wr, traj_data=(traj3_x, traj3_y), bp_layer=None, iteration=it3)

for _ in range(6):
    render_frame(k_quad, w1, w2, w3, wr, bp_layer=None, iteration=it3)

# -----------------------------------------------------------------------------
# FINAL STATE: Strictly Decreasing Lyapunov Function & Full Trajectory Family
# -----------------------------------------------------------------------------
n_final = 42
for f in range(n_final):
    frac = min(1.0, (f + 1) / 35.0)
    render_frame(k_quad, w1, w2, w3, wr, final_bundle_frac=frac, 
                 chk_active=False, fwd_layer=None, bp_layer=None, iteration=it3)

for _ in range(25):
    render_frame(k_quad, w1, w2, w3, wr, final_bundle_frac=1.0, 
                 chk_active=False, fwd_layer=None, bp_layer=None, iteration=it3)

writer.close()
plt.close(fig)
print("Animation completed successfully: cegis_concept_2d.mp4")
