"""
2D Conceptual Animation: Adversarial Replay Buffer Pipeline
------------------------------------------------------------
Minimalist, publication-style 2D animation of the Adversarial Replay Buffer:
- Pure mathematical symbols only: \mathcal{S}, \mathcal{C}, \mathcal{L}_{cond}, \mathcal{B}_\rho
- No headers, no titles, no descriptive text, no tau annotations
- Buffer points stay in the buffer during sampling (sampling without removing)
- Random points in \mathcal{S} strictly avoid the label position
- CEX points in \mathcal{C} have a 2-step lifespan (age 0 = bright red, age 1 = deep crimson, age >= 2 = evicted)
- Sample particles travel along curved conduit tracks into the mini-batch \mathcal{B}_\rho of \mathcal{L}_{cond}
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

# -----------------------------------------------------------------------------
# 1. Thesis Color Palette (matching pgfplotssetup.tex & defence)
# -----------------------------------------------------------------------------
COLOR_NAVY       = "#1B365D"   # Primary thesis blue
COLOR_TEAL       = "#0D9488"   # Secondary: Explorative state pool S
COLOR_AMBER      = "#D97706"   # Replay buffer frame / CEX accent
COLOR_CEX_FRESH  = "#DC2626"   # Fresh CEX (bright red)
COLOR_CEX_AGED   = "#991B1B"   # Retained CEX 1 step old (deep crimson)
COLOR_DARK       = "#1F2937"   # Structural elements & labels
COLOR_TRACK      = "#CBD5E1"   # Conduit track line
COLOR_BG_S       = "#F0FDFA"   # Soft teal background
COLOR_BG_C       = "#FFFBEB"   # Soft amber background
COLOR_BG_LOSS    = "#F8FAFC"   # Soft slate background

# -----------------------------------------------------------------------------
# 2. Bézier Curve Helper for Smooth Particle Flow
# -----------------------------------------------------------------------------
def cubic_bezier(p0, p1, p2, p3, t):
    """Computes point on cubic bezier curve at parameter t in [0, 1]."""
    return (
        (1 - t)**3 * p0
        + 3 * (1 - t)**2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )

# Routing tracks
# Upper track: from S exit (4.8, 4.35) to Loss entry (7.4, 3.20)
p_track_s0 = np.array([4.8, 4.35])
p_track_s1 = np.array([5.8, 4.35])
p_track_s2 = np.array([6.4, 3.20])
p_track_s3 = np.array([7.4, 3.20])

# Lower track: from C exit (4.8, 1.85) to Loss entry (7.4, 2.50)
p_track_c0 = np.array([4.8, 1.85])
p_track_c1 = np.array([5.8, 1.85])
p_track_c2 = np.array([6.4, 2.50])
p_track_c3 = np.array([7.4, 2.50])

# Track points for drawing static guide tracks
t_vals = np.linspace(0, 1, 60)
track_s_pts = np.array([cubic_bezier(p_track_s0, p_track_s1, p_track_s2, p_track_s3, t) for t in t_vals])
track_c_pts = np.array([cubic_bezier(p_track_c0, p_track_c1, p_track_c2, p_track_c3, t) for t in t_vals])

# -----------------------------------------------------------------------------
# 3. State & Counterexample Point Generation (Excluding Text Zones)
# -----------------------------------------------------------------------------
np.random.seed(42)

def generate_s_points(n=24):
    """
    Generates random positions in Box S: x in [1.05, 4.60], y in [3.50, 5.15].
    Strictly excludes the top-left area where $\mathcal{S}$ is displayed:
    Exclusion box: x in [0.85, 1.95] and y in [4.45, 5.30].
    """
    pts = []
    while len(pts) < n:
        rx = np.random.uniform(1.05, 4.58)
        ry = np.random.uniform(3.52, 5.18)
        # Avoid label area
        if rx < 1.95 and ry > 4.45:
            continue
        pts.append((rx, ry))
    return pts

# Pre-generate 3 distinct random sets for the 3 steps
s_pts_step1 = generate_s_points(24)
s_pts_step2 = generate_s_points(24)
s_pts_step3 = generate_s_points(24)

# Counterexamples in Box C (x in [1.05, 4.60], y in [1.00, 2.65])
# Label $\mathcal{C}$ is at top-left: x in [0.85, 1.95], y in [1.95, 2.80]
# All CEX coordinates strictly lie outside the label area
cex_batch1 = [(2.15, 2.45), (2.85, 1.35), (3.65, 2.25), (4.30, 1.45)]
cex_batch2 = [(2.25, 1.45), (3.10, 2.35), (3.80, 1.30), (4.35, 2.30)]
cex_batch3 = [(2.05, 1.85), (2.75, 2.40), (3.45, 1.45), (4.20, 1.80)]

# Indices of S points sampled for streaming
stream_s_indices = [2, 7, 11, 16, 21]

# Destinations inside Mini-Batch box B_rho (x in [7.75, 9.25], y in [2.40, 2.95])
mb_s_destinations = [(7.85, 2.80), (8.15, 2.90), (8.50, 2.80), (8.85, 2.90), (9.15, 2.80)]
mb_c_destinations = [(8.05, 2.50), (8.50, 2.50), (8.95, 2.50)]

# Slide background color matching the right PowerPoint card
SLIDE_BG         = "#D9D9D9"   # Right card background (or set to "#F2F2F2" if unified)

# -----------------------------------------------------------------------------
# 4. Matplotlib Setup (Compact Bounds for Card)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=100)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
output_filename = "adversarial_buffer.mp4"
writer = imageio.get_writer(output_filename, fps=30, quality=9)
print("Rendering Minimalist Adversarial Buffer Animation (MP4)...")

def draw_layout(highlight_loss=False):
    """Draws background layout with ONLY S, C, L_cond, and B_rho."""
    ax.clear()
    ax.set_xlim(0.48, 9.80)
    ax.set_ylim(0.50, 5.75)
    ax.axis("off")
    fig.patch.set_facecolor(SLIDE_BG)
    ax.set_facecolor(SLIDE_BG)

    # Outer Container: Adversarial Replay Buffer Frame (dashed amber line)
    outer_box = patches.FancyBboxPatch(
        (0.65, 0.65), 4.35, 4.90,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        edgecolor=COLOR_AMBER, facecolor=SLIDE_BG,
        linestyle="--", linewidth=1.6
    )
    ax.add_patch(outer_box)

    # Box 1 (Top): Explorative State Buffer S
    box_s = patches.FancyBboxPatch(
        (0.85, 3.35), 3.95, 1.95,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        edgecolor=COLOR_TEAL, facecolor=COLOR_BG_S,
        linewidth=1.6
    )
    ax.add_patch(box_s)
    # ONLY symbol S
    ax.text(
        1.35, 4.88, r"$\mathcal{S}$",
        fontsize=22, fontweight="bold", color=COLOR_TEAL, ha="center", va="center"
    )

    # Box 2 (Bottom): Counterexample Buffer C
    box_c = patches.FancyBboxPatch(
        (0.85, 0.85), 3.95, 1.95,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        edgecolor=COLOR_AMBER, facecolor=COLOR_BG_C,
        linewidth=1.6
    )
    ax.add_patch(box_c)
    # ONLY symbol C
    ax.text(
        1.35, 2.38, r"$\mathcal{C}$",
        fontsize=22, fontweight="bold", color=COLOR_AMBER, ha="center", va="center"
    )

    # Conduit Tracks (Bahnen) with sampling fraction labels
    # Upper track
    ax.plot(track_s_pts[:, 0], track_s_pts[:, 1], color=COLOR_TRACK, linewidth=3.5, zorder=2)
    ax.plot(track_s_pts[:, 0], track_s_pts[:, 1], color="white", linewidth=1.5, zorder=2)
    ax.text(
        6.05, 4.48, r"$1 - r_{\mathcal{C}}$",
        fontsize=12, fontweight="bold", color=COLOR_TEAL, ha="center", va="bottom"
    )
    ax.annotate(
        "", xy=(6.50, 3.45), xytext=(6.20, 3.85),
        arrowprops=dict(arrowstyle="->", color=COLOR_TEAL, lw=1.6)
    )

    # Lower track
    ax.plot(track_c_pts[:, 0], track_c_pts[:, 1], color=COLOR_TRACK, linewidth=3.5, zorder=2)
    ax.plot(track_c_pts[:, 0], track_c_pts[:, 1], color="white", linewidth=1.5, zorder=2)
    ax.text(
        6.05, 1.68, r"$r_{\mathcal{C}}$",
        fontsize=12, fontweight="bold", color=COLOR_CEX_FRESH, ha="center", va="top"
    )
    ax.annotate(
        "", xy=(6.50, 2.35), xytext=(6.20, 1.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_CEX_FRESH, lw=1.6)
    )

    # Target Block: L_cond
    loss_edge_color = COLOR_NAVY if not highlight_loss else COLOR_AMBER
    loss_line_w = 2.0 if not highlight_loss else 2.6
    loss_bg = COLOR_BG_LOSS if not highlight_loss else "#FEF3C7"
    
    box_loss = patches.FancyBboxPatch(
        (7.40, 1.45), 2.25, 3.30,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        edgecolor=loss_edge_color, facecolor=loss_bg,
        linewidth=loss_line_w
    )
    ax.add_patch(box_loss)

    # ONLY symbol L_cond
    ax.text(
        8.525, 4.15, r"$\mathcal{L}_{\mathrm{cond}}$",
        fontsize=21, fontweight="bold", color=COLOR_NAVY, ha="center", va="center"
    )

    # Mini-batch container B_rho inside loss block
    batch_box = patches.FancyBboxPatch(
        (7.65, 1.70), 1.75, 1.60,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        edgecolor="#94A3B8", facecolor="white",
        linestyle=":", linewidth=1.4
    )
    ax.add_patch(batch_box)
    # ONLY symbol B_rho
    ax.text(
        8.525, 1.98, r"$\mathcal{B}_\rho$",
        fontsize=15, fontweight="bold", color="#64748B", ha="center", va="center"
    )

def render_frame_to_writer():
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    writer.append_data(rgba[:h - h%2, :w - w%2, :3])

def stream_particles(track_pts, start_pos, target_pos, progress):
    """Interpolates particle moving from inside box -> track -> loss mini-batch."""
    if progress <= 0.22:
        t_sub = progress / 0.22
        pos = (1 - t_sub) * np.array(start_pos) + t_sub * track_pts[0]
    elif progress <= 0.82:
        t_sub = (progress - 0.22) / 0.60
        idx = int(t_sub * (len(track_pts) - 1))
        pos = track_pts[idx]
    else:
        t_sub = (progress - 0.82) / 0.18
        pos = (1 - t_sub) * track_pts[-1] + t_sub * np.array(target_pos)
    return pos

# =============================================================================
# SEQUENCE EXECUTION
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: Random S_1 drawn + CEX Batch 1 arrives -> Stream to B_rho
# -----------------------------------------------------------------------------
# S_1 appears
for f in range(16):
    draw_layout()
    alpha = min(1.0, (f + 1) / 10.0)
    for px, py in s_pts_step1:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, alpha=alpha, zorder=4)
    render_frame_to_writer()

# CEX Batch 1 arrives (fresh = bright red, NO tau text)
for f in range(18):
    draw_layout()
    for px, py in s_pts_step1:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    alpha = min(1.0, (f + 1) / 10.0)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, alpha=alpha, zorder=5)
    render_frame_to_writer()

# Stream particles along tracks into B_rho
# CRITICAL: Points in S and C DO NOT disappear from buffer!
n_stream = 36
for f in range(n_stream):
    draw_layout()
    prog = (f + 1) / n_stream

    # ALL points remain visible in S buffer!
    for px, py in s_pts_step1:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    # Highlight sampled points in S
    for s_idx in stream_s_indices:
        ax.scatter(s_pts_step1[s_idx][0], s_pts_step1[s_idx][1], s=140, facecolors="none", edgecolors=COLOR_TEAL, linewidth=1.5, linestyle="--", zorder=6)

    # ALL points remain visible in C buffer!
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    # Highlight sampled points in C
    for i in range(2):
        ax.scatter(cex_batch1[i][0], cex_batch1[i][1], s=230, facecolors="none", edgecolors=COLOR_CEX_FRESH, linewidth=1.5, linestyle="--", zorder=6)

    # Moving copy from S
    for i, s_idx in enumerate(stream_s_indices):
        curr_p = stream_particles(track_s_pts, s_pts_step1[s_idx], mb_s_destinations[i], prog)
        ax.scatter(curr_p[0], curr_p[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)

    # Moving copy from C (2 counterexamples)
    for i in range(2):
        curr_p = stream_particles(track_c_pts, cex_batch1[i], mb_c_destinations[i], prog)
        ax.scatter(curr_p[0], curr_p[1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)

    render_frame_to_writer()

# Loss evaluates (subtle pulse)
for f in range(15):
    draw_layout(highlight_loss=(f < 10))
    for px, py in s_pts_step1:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    # Arrived batch in B_rho
    for dest in mb_s_destinations:
        ax.scatter(dest[0], dest[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)
    for dest in mb_c_destinations[:2]:
        ax.scatter(dest[0], dest[1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)
    render_frame_to_writer()

# -----------------------------------------------------------------------------
# STEP 2: S_2 randomly redrawn | Batch 1 RETAINED (aged) | Batch 2 arrives -> Stream
# -----------------------------------------------------------------------------
# S_1 dissolves, S_2 spawns randomly | Batch 1 transitions to deeper red (retained!)
for f in range(22):
    draw_layout()
    alpha_out = max(0.0, 1.0 - (f + 1) / 10.0)
    alpha_in = min(1.0, max(0.0, (f - 6) / 10.0))
    
    if alpha_out > 0:
        for px, py in s_pts_step1:
            ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, alpha=alpha_out, zorder=4)
    if alpha_in > 0:
        for px, py in s_pts_step2:
            ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, alpha=alpha_in, zorder=4)

    # Batch 1 in C buffer transitions to aged crimson (STAYS IN BUFFER!)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
        
    render_frame_to_writer()

# Batch 2 arrives into C (fresh bright red) while Batch 1 STAYS
for f in range(18):
    draw_layout()
    for px, py in s_pts_step2:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
        
    # Batch 1 (aged, stays in buffer)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
        
    # Batch 2 (fresh, appears)
    alpha_b2 = min(1.0, (f + 1) / 10.0)
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, alpha=alpha_b2, zorder=5)
        
    render_frame_to_writer()

# Stream particles: buffer points STAY!
for f in range(n_stream):
    draw_layout()
    prog = (f + 1) / n_stream

    # ALL points remain in S!
    for px, py in s_pts_step2:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    # Highlight sampled points in S
    for s_idx in stream_s_indices:
        ax.scatter(s_pts_step2[s_idx][0], s_pts_step2[s_idx][1], s=140, facecolors="none", edgecolors=COLOR_TEAL, linewidth=1.5, linestyle="--", zorder=6)

    # ALL points remain in C (both batches stay in buffer!)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    # Highlight sampled points in C
    ax.scatter(cex_batch1[1][0], cex_batch1[1][1], s=230, facecolors="none", edgecolors=COLOR_CEX_AGED, linewidth=1.5, linestyle="--", zorder=6)
    ax.scatter(cex_batch2[2][0], cex_batch2[2][1], s=230, facecolors="none", edgecolors=COLOR_CEX_FRESH, linewidth=1.5, linestyle="--", zorder=6)

    # Moving S particles
    for i, s_idx in enumerate(stream_s_indices):
        curr_p = stream_particles(track_s_pts, s_pts_step2[s_idx], mb_s_destinations[i], prog)
        ax.scatter(curr_p[0], curr_p[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)

    # Moving CEX particles (one from Batch 1 aged, one from Batch 2 fresh)
    p_c1 = stream_particles(track_c_pts, cex_batch1[1], mb_c_destinations[0], prog)
    ax.scatter(p_c1[0], p_c1[1], s=95, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=7)
    p_c2 = stream_particles(track_c_pts, cex_batch2[2], mb_c_destinations[1], prog)
    ax.scatter(p_c2[0], p_c2[1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)

    render_frame_to_writer()

# Loss evaluates
for f in range(15):
    draw_layout(highlight_loss=(f < 10))
    for px, py in s_pts_step2:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    for cx, cy in cex_batch1:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    for dest in mb_s_destinations:
        ax.scatter(dest[0], dest[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)
    ax.scatter(mb_c_destinations[0][0], mb_c_destinations[0][1], s=95, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=7)
    ax.scatter(mb_c_destinations[1][0], mb_c_destinations[1][1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)
    render_frame_to_writer()

# -----------------------------------------------------------------------------
# STEP 3: S_3 randomly redrawn | Batch 1 EXPIRES & dissolves | Batch 2 aged | Batch 3 arrives
# -----------------------------------------------------------------------------
# Batch 1 expires (dissolves / disappears from buffer) | Batch 2 shifts to aged crimson | S_3 spawns
for f in range(24):
    draw_layout()
    # S_3 appears randomly
    alpha_s3 = min(1.0, (f + 1) / 10.0)
    for px, py in s_pts_step3:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, alpha=alpha_s3, zorder=4)

    # Batch 1 expires: shrinks and dissolves from buffer
    alpha_exp = max(0.0, 1.0 - (f + 1) / 16.0)
    if alpha_exp > 0:
        for cx, cy in cex_batch1:
            ax.scatter(cx, cy, s=115 * alpha_exp, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.0, alpha=alpha_exp, zorder=5)

    # Batch 2 is retained and ages to crimson
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)

    render_frame_to_writer()

# Batch 3 arrives into C (fresh bright red) while Batch 2 STAYS and Batch 1 is GONE
for f in range(18):
    draw_layout()
    for px, py in s_pts_step3:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
        
    # Batch 2 (aged, stays)
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
        
    # Batch 3 (fresh, appears)
    alpha_b3 = min(1.0, (f + 1) / 10.0)
    for cx, cy in cex_batch3:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, alpha=alpha_b3, zorder=5)
        
    render_frame_to_writer()

# Stream particles into Loss
for f in range(n_stream):
    draw_layout()
    prog = (f + 1) / n_stream

    # ALL points remain in S!
    for px, py in s_pts_step3:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    # Highlight sampled points in S
    for s_idx in stream_s_indices:
        ax.scatter(s_pts_step3[s_idx][0], s_pts_step3[s_idx][1], s=140, facecolors="none", edgecolors=COLOR_TEAL, linewidth=1.5, linestyle="--", zorder=6)

    # Points in C remain (Batch 2 and Batch 3)!
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
    for cx, cy in cex_batch3:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    # Highlight sampled points in C
    ax.scatter(cex_batch2[0][0], cex_batch2[0][1], s=230, facecolors="none", edgecolors=COLOR_CEX_AGED, linewidth=1.5, linestyle="--", zorder=6)
    ax.scatter(cex_batch3[1][0], cex_batch3[1][1], s=230, facecolors="none", edgecolors=COLOR_CEX_FRESH, linewidth=1.5, linestyle="--", zorder=6)

    # Moving S particles
    for i, s_idx in enumerate(stream_s_indices):
        curr_p = stream_particles(track_s_pts, s_pts_step3[s_idx], mb_s_destinations[i], prog)
        ax.scatter(curr_p[0], curr_p[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)

    # Moving CEX particles (one from Batch 2 aged, one from Batch 3 fresh)
    p_c2 = stream_particles(track_c_pts, cex_batch2[0], mb_c_destinations[0], prog)
    ax.scatter(p_c2[0], p_c2[1], s=95, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=7)
    p_c3 = stream_particles(track_c_pts, cex_batch3[1], mb_c_destinations[1], prog)
    ax.scatter(p_c3[0], p_c3[1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)

    render_frame_to_writer()

# Loss evaluates (hold final state)
for f in range(30):
    draw_layout(highlight_loss=(f < 10))
    for px, py in s_pts_step3:
        ax.scatter(px, py, s=50, color=COLOR_TEAL, edgecolors="white", linewidth=0.8, zorder=4)
    for cx, cy in cex_batch2:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=5)
    for cx, cy in cex_batch3:
        ax.scatter(cx, cy, s=115, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=5)
    for dest in mb_s_destinations:
        ax.scatter(dest[0], dest[1], s=65, color=COLOR_TEAL, edgecolors="white", linewidth=1.0, zorder=7)
    ax.scatter(mb_c_destinations[0][0], mb_c_destinations[0][1], s=95, color=COLOR_CEX_AGED, edgecolors="white", linewidth=1.2, zorder=7)
    ax.scatter(mb_c_destinations[1][0], mb_c_destinations[1][1], s=95, color=COLOR_CEX_FRESH, edgecolors="white", linewidth=1.2, zorder=7)
    render_frame_to_writer()

writer.close()
plt.close(fig)
print("Animation completed successfully: adversarial_buffer.mp4")
