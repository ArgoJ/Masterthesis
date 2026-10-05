import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

COLOR_NAVY      = "#1B365D"
COLOR_SUBLEVEL  = "#D97706"
COLOR_CEX       = "#DC2626"
COLOR_DARK      = "#1F2937"
COLOR_TEAL      = "#0D9488"
SLIDE_BG        = "#FFFFFF"

fig = plt.figure(figsize=(9.6, 5.44), dpi=100)
gs = fig.add_gridspec(1, 2, width_ratios=[0.90, 1.10], left=0.03, right=0.97, bottom=0.06, top=0.94, wspace=0.08)

ax_nn = fig.add_subplot(gs[0, 0])
ax_plot = fig.add_subplot(gs[0, 1])

fig.patch.set_facecolor(SLIDE_BG)
ax_nn.set_facecolor(SLIDE_BG)
ax_plot.set_facecolor(SLIDE_BG)

ax_plot.set_xlim(-2.25, 2.25)
ax_plot.set_ylim(-2.25, 2.25)
ax_plot.set_aspect("equal")

ax_plot.spines["top"].set_visible(False)
ax_plot.spines["right"].set_visible(False)
ax_plot.spines["left"].set_position("zero")
ax_plot.spines["bottom"].set_position("zero")
ax_plot.spines["left"].set_color(COLOR_DARK)
ax_plot.spines["bottom"].set_color(COLOR_DARK)
ax_plot.spines["left"].set_linewidth(1.1)
ax_plot.spines["bottom"].set_linewidth(1.1)
ax_plot.set_xticks([-2, -1, 1, 2])
ax_plot.set_yticks([-2, -1, 1, 2])
ax_plot.tick_params(colors=COLOR_DARK, labelsize=9.0, width=1.0)
ax_plot.text(2.18, -0.02, r"$x_1$", fontsize=12, color=COLOR_DARK, va="top", ha="left")
ax_plot.text(-0.02, 2.18, r"$x_2$", fontsize=12, color=COLOR_DARK, va="bottom", ha="right")

circ = patches.Circle((0, 0), 1.2, facecolor=COLOR_SUBLEVEL, alpha=0.25, edgecolor=COLOR_SUBLEVEL, linewidth=2.0)
ax_plot.add_patch(circ)
ax_plot.scatter(0, 0, s=28, color=COLOR_DARK, zorder=6)
ax_plot.text(-0.08, -0.15, r"$x^\star$", fontsize=9.5, color=COLOR_DARK)

ax_nn.set_xlim(0, 1)
ax_nn.set_ylim(0, 1)
ax_nn.axis("off")

layer_counts = [2, 4, 3, 1]
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

def draw_network_simple(ax, node_coords, title, out_name, accent_color, node_fill):
    y_title = node_coords[1][-1][1] + 0.055
    ax.text(0.26, y_title, title, fontsize=15.0, color=accent_color, va="center", ha="center")
    
    rng = np.random.RandomState(42)
    for g in range(3):
        W = rng.uniform(0.85, 1.45, size=(len(node_coords[g]), len(node_coords[g+1])))
        for i, (x1, y1) in enumerate(node_coords[g]):
            for j, (x2, y2) in enumerate(node_coords[g+1]):
                w = W[i, j]
                ax.plot([x1, x2], [y1, y2], color=accent_color, linewidth=w * 1.15, alpha=0.48, zorder=1)
                
    for l_idx, coords in enumerate(node_coords):
        for n_idx, (nx, ny) in enumerate(coords):
            ax.scatter(nx, ny, s=145, facecolor=node_fill, edgecolors=accent_color, linewidth=1.5, zorder=5)
            if l_idx == 0:
                inp_lbl = r"$x_1$" if n_idx == 1 else r"$x_2$"
                ax.text(nx - 0.040, ny, inp_lbl, fontsize=12.5, color=COLOR_DARK, va="center", ha="right", zorder=6)
            elif l_idx == len(node_coords) - 1:
                ax.text(nx + 0.040, ny, out_name, fontsize=13.5, color=COLOR_DARK, va="center", ha="left", zorder=6)

draw_network_simple(ax_nn, nodes_pi, title=r"$\pi_\theta(x)$", out_name=r"$u$", accent_color=COLOR_TEAL, node_fill="#E6FFFA")
draw_network_simple(ax_nn, nodes_v,  title=r"$V_\phi(x)$",  out_name=r"$V$", accent_color=COLOR_SUBLEVEL, node_fill="#FFFBEB")

# Condition block position (wide pill)
chk_x = 0.73
chk_y = 0.50
chk_w = 0.48
chk_h = 0.14

offset = 0.12
ax_nn.annotate("", xy=(chk_x, chk_y + chk_h/2), xytext=(xs[-1] + offset, 0.74),
                arrowprops=dict(arrowstyle="->", color=COLOR_TEAL, lw=1.6, connectionstyle="arc3,rad=0.4"))
ax_nn.annotate("", xy=(chk_x, chk_y - chk_h/2), xytext=(xs[-1] + offset, 0.26),
                arrowprops=dict(arrowstyle="->", color=COLOR_SUBLEVEL, lw=1.6, connectionstyle="arc3,rad=-0.4"))

# Condition check pill
pill = patches.FancyBboxPatch(
    (chk_x - chk_w/2, chk_y - chk_h/2), chk_w, chk_h,
    boxstyle="round,pad=0.015,rounding_size=0.035",
    facecolor="#FEF2F2", edgecolor=COLOR_CEX, linewidth=1.6, zorder=6
)
ax_nn.add_patch(pill)

ax_nn.text(chk_x, chk_y + 0.026, r"$V_\phi(x^+) > (1-\kappa)V_\phi(x)$", 
           fontsize=12.0, color=COLOR_CEX, va="center", ha="center", zorder=7)
ax_nn.text(chk_x, chk_y - 0.026, r"$x^+ = f(x, u)$", 
           fontsize=11.0, color="#64748B", va="center", ha="center", zorder=7)

# Compute start point of flight in ax_plot coordinates
fig.canvas.draw()
inv_plot = ax_plot.transData.inverted()
p_src_disp = ax_nn.transData.transform((chk_x + chk_w/2, chk_y))
p_src_plot = inv_plot.transform(p_src_disp)

cx_target, cy_target = -0.6, 0.4
t_flight = 0.55
curr_x = (1 - t_flight) * p_src_plot[0] + t_flight * cx_target
curr_y = (1 - t_flight) * p_src_plot[1] + t_flight * cy_target + 0.35 * np.sin(t_flight * np.pi)

# Scatter with clip_on=False: perfectly circular, no distortion!
ax_plot.scatter(curr_x, curr_y, s=75, facecolor=COLOR_CEX, edgecolors="white", linewidth=1.1, clip_on=False, zorder=12)

plt.savefig("/home/josua/programming_stuff/projects/ma-tex/animations/test_frame.png", dpi=100)
print("Saved test_frame.png with wide pill and large fonts")
