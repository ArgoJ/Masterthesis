"""
CEGIS (Counterexample-Guided Inductive Synthesis) Concept Animation
-------------------------------------------------------------------
Demonstrates the 3-step CEGIS loop for Lyapunov function synthesis and 
Region of Attraction (ROA) sublevel set expansion:

- Background: pure white
- No grid, no title or text overlays
- Resolution: 960x540 (compact for presentation slides)
- Clear elevated 3/4 perspective looking into the Lyapunov basin
- No box frames or borders: the sublevel set expands directly up to the edge of the Lyapunov function
- rho is ALWAYS the smallest boundary value of the Lyapunov function domain: rho = min_{x in d D} V(x)
- Sublevel set is present from the very beginning in red
- In the sublevel set, counterexamples (cex) are found on a deformed surface
- Test trajectories show violation / get trapped
- The cex are resolved by inductive synthesis (surface reshaped, trajectory reaches origin)
- Trajectories are removed after each optimization to keep the scene clean
- As the learner shapes and steepens the Lyapunov basin, boundary values rise and rho grows slowly up to the boundary of the function
- New cex appear in the newly encompassed area, which are then resolved
- Total of 3 iterative CEGIS steps until a strictly decreasing Lyapunov function stands
- All trajectories are continuous solid lines (tubes)
"""

import numpy as np
import pyvista as pv

# -----------------------------------------------------------------------------
# 1. State Space Grid & Surface Mathematics
# -----------------------------------------------------------------------------
n_grid = 120
x_vals = np.linspace(-2.0, 2.0, n_grid)
y_vals = np.linspace(-2.0, 2.0, n_grid)
X, Y = np.meshgrid(x_vals, y_vals)

# Boundary of the Lyapunov function domain
b_domain = 1.95

# Defect centers for 3 concentric zones / steps
# Step 1: inner zone (r ~ 0.5 - 0.75)
ce_step1 = [
    (0.60, 0.40),
    (-0.55, 0.45),
    (0.35, -0.60),
    (-0.50, -0.40),
]

# Step 2: middle zone (r ~ 0.95 - 1.20)
ce_step2 = [
    (1.05, 0.30),
    (-0.90, 0.70),
    (0.65, -1.00),
    (-1.00, -0.50),
    (0.15, 1.15),
]

# Step 3: outer zone (r ~ 1.30 - 1.55)
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

# Dynamic Lyapunov function parameters
k_quad = 0.075
w1, w2, w3, wr = 1.0, 1.0, 1.0, 1.0

def current_v(x, y):
    base = k_quad * (x**2 + y**2) + 0.015 * (x**4 + y**4) / 4.0
    val = base + w1*defect_field_1(x, y) + w2*defect_field_2(x, y) + w3*defect_field_3(x, y) + wr*ripple_field(x, y)
    return np.maximum(val, 0.015)

def get_domain_min_boundary_val(v_func, b, n_edge=60):
    """Computes rho = min_{x in d D} V(x) on the boundary of the Lyapunov function domain."""
    edge = np.linspace(-b, b, n_edge)
    v_top = v_func(edge, np.full_like(edge, b))
    v_bottom = v_func(edge, np.full_like(edge, -b))
    v_right = v_func(np.full_like(edge, b), edge)
    v_left = v_func(np.full_like(edge, -b), edge)
    return float(np.min([np.min(v_top), np.min(v_bottom), np.min(v_right), np.min(v_left)]))

# -----------------------------------------------------------------------------
# 2. Dynamics & Trajectory Simulation
# -----------------------------------------------------------------------------
def simulate_trajectory(x0, y0, t_max=2.5, steps=90):
    """Simulates nonlinear spiral dynamics dx/dt = f(x)."""
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

def build_tube(x_pts, y_pts, v_func, z_offset=0.025, radius=0.018):
    z_pts = [float(v_func(np.array([xi]), np.array([yi]))[0]) + z_offset for xi, yi in zip(x_pts, y_pts)]
    pts = np.column_stack((x_pts, y_pts, z_pts))
    return pv.lines_from_points(pts).tube(radius=radius)

# Diagnostic test trajectories for each iteration
traj1_x, traj1_y = simulate_trajectory(1.10, 0.15, t_max=2.0, steps=70)
traj2_x, traj2_y = simulate_trajectory(-1.35, 0.75, t_max=2.0, steps=70)
traj3_x, traj3_y = simulate_trajectory(1.60, -0.50, t_max=2.2, steps=75)

# Final trajectory bundle
final_trajs_xy = []
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    r_start = 1.55
    tx, ty = simulate_trajectory(r_start * np.cos(angle), r_start * np.sin(angle), t_max=3.0, steps=85)
    final_trajs_xy.append((tx, ty))

# -----------------------------------------------------------------------------
# 3. PyVista Scene Setup (Clean, No Box Frame, No Grid, No Title)
# -----------------------------------------------------------------------------
Z_init = current_v(X, Y)
grid = pv.StructuredGrid(X, Y, Z_init)
grid.point_data["V"] = Z_init.flatten(order="F")

plotter = pv.Plotter(off_screen=True, window_size=[960, 540])
plotter.set_background("white")

# Main surface
plotter.add_mesh(
    grid,
    scalars="V",
    cmap="Blues",
    smooth_shading=True,
    opacity=0.88,
    show_scalar_bar=False,
    specular=0.25,
    name="surface"
)

# Origin marker
origin_marker = pv.Sphere(radius=0.05, center=(0, 0, 0.02))
plotter.add_mesh(origin_marker, color="#16a34a", name="origin")

# Elevated camera perspective looking down into the basin
plotter.camera_position = [(5.0, -5.4, 5.8), (0.0, 0.0, 0.40), (0.0, 0.0, 1.0)]

def step_cam(angle_deg=0.08):
    plotter.camera.azimuth += angle_deg

def update_surface():
    Z_new = current_v(X, Y)
    grid.points[:, 2] = Z_new.flatten(order="F")
    grid.point_data["V"] = Z_new.flatten(order="F")

def update_sublevel():
    # rho is strictly computed as min_{x in d D} V(x) at the edge of the Lyapunov function
    rho_val = get_domain_min_boundary_val(current_v, b_domain)
    
    # Red contour at V(x) = rho
    contour = grid.contour(isosurfaces=[rho_val], scalars="V")
    if contour.n_points > 0:
        plotter.add_mesh(contour.tube(radius=0.022), color="#dc2626", name="sublevel_contour")
    else:
        plotter.remove_actor("sublevel_contour")
        
    # Red translucent filled sublevel set region V(x) <= rho
    sublevel = grid.clip_scalar(value=rho_val, invert=True, scalars="V")
    if sublevel.n_points > 0:
        sublevel.points[:, 2] += 0.008
        plotter.add_mesh(sublevel, color="#dc2626", opacity=0.38, name="sublevel_fill")
    else:
        plotter.remove_actor("sublevel_fill")
        
    return rho_val

# Open movie
plotter.open_movie("lyapunov_learning_cegis.mp4", framerate=30, quality=9)
print("Rendering CEGIS animation...")

# Sublevel set is present from the very beginning at rho = min_{d D} V
update_sublevel()

# Trajectory 1 enters and gets trapped near dent
for f in range(25):
    idx = max(2, int((f + 1) / 25 * 35))
    tube1 = build_tube(traj1_x[:idx], traj1_y[:idx], current_v)
    plotter.add_mesh(tube1, color="#0284c7", name="traj1")
    step_cam(0.08)
    plotter.write_frame()

# =============================================================================
# ITERATION 1: Counterexamples in sublevel set -> Resolve & Remove Trajectory
# =============================================================================
# 1. Counterexamples appear
for s in np.linspace(0.02, 0.075, 10):
    for i, (cx, cy) in enumerate(ce_step1):
        cz = float(current_v(np.array([cx]), np.array([cy]))[0])
        sp = pv.Sphere(radius=s, center=(cx, cy, cz))
        plotter.add_mesh(sp, color="#dc2626", name=f"ce1_{i}")
    step_cam(0.08)
    plotter.write_frame()

for _ in range(15):
    step_cam(0.08)
    plotter.write_frame()

# 2. Inductive synthesis: w1 morphs to 0 (dents iron out in zone 1)
n_m1 = 30
for step in range(n_m1):
    alpha = (step + 1) / n_m1
    w1 = 1.0 - alpha
    wr = 1.0 - 0.33 * alpha
    update_surface()
    update_sublevel()
    
    # Cex 1 shrink and dissolve
    scale = max(0.075 * (1.0 - alpha), 0.001)
    for i, (cx, cy) in enumerate(ce_step1):
        if alpha < 0.95:
            cz = float(current_v(np.array([cx]), np.array([cy]))[0])
            sp = pv.Sphere(radius=scale, center=(cx, cy, cz))
            plotter.add_mesh(sp, color="#dc2626", name=f"ce1_{i}")
        else:
            plotter.remove_actor(f"ce1_{i}")
            
    # Trajectory 1 adapts dynamically and flows down to origin
    idx = max(2, int(35 + alpha * 35))
    tube1 = build_tube(traj1_x[:idx], traj1_y[:idx], current_v)
    plotter.add_mesh(tube1, color="#0284c7", name="traj1")
    
    step_cam(0.08)
    plotter.write_frame()

for i in range(len(ce_step1)):
    plotter.remove_actor(f"ce1_{i}")

# Trajectory 1 is removed after optimization
plotter.remove_actor("traj1")

for _ in range(8):
    step_cam(0.08)
    plotter.write_frame()

# 3. Learner reshapes basin: k_quad increases from 0.075 to 0.155
# This raises boundary values of the function, and rho = min_{d D} V(x) grows smoothly!
n_grow1 = 45
k_start1, k_end1 = 0.075, 0.155
for step in range(n_grow1):
    k_quad = k_start1 + (k_end1 - k_start1) * (step + 1) / n_grow1
    update_surface()
    update_sublevel()
    step_cam(0.08)
    plotter.write_frame()

# =============================================================================
# ITERATION 2: New Counterexamples in expanded sublevel set -> Resolve & Remove
# =============================================================================
# Trajectory 2 approaches middle zone defect
for f in range(20):
    idx = max(2, int((f + 1) / 20 * 38))
    tube2 = build_tube(traj2_x[:idx], traj2_y[:idx], current_v)
    plotter.add_mesh(tube2, color="#0284c7", name="traj2")
    step_cam(0.08)
    plotter.write_frame()

# 1. New counterexamples appear in expanded region
for s in np.linspace(0.02, 0.075, 10):
    for i, (cx, cy) in enumerate(ce_step2):
        cz = float(current_v(np.array([cx]), np.array([cy]))[0])
        sp = pv.Sphere(radius=s, center=(cx, cy, cz))
        plotter.add_mesh(sp, color="#dc2626", name=f"ce2_{i}")
    step_cam(0.08)
    plotter.write_frame()

for _ in range(15):
    step_cam(0.08)
    plotter.write_frame()

# 2. Inductive synthesis: w2 morphs to 0
n_m2 = 30
for step in range(n_m2):
    alpha = (step + 1) / n_m2
    w2 = 1.0 - alpha
    wr = 0.67 - 0.33 * alpha
    update_surface()
    update_sublevel()
    
    # Cex 2 dissolve
    scale = max(0.075 * (1.0 - alpha), 0.001)
    for i, (cx, cy) in enumerate(ce_step2):
        if alpha < 0.95:
            cz = float(current_v(np.array([cx]), np.array([cy]))[0])
            sp = pv.Sphere(radius=scale, center=(cx, cy, cz))
            plotter.add_mesh(sp, color="#dc2626", name=f"ce2_{i}")
        else:
            plotter.remove_actor(f"ce2_{i}")
            
    # Trajectory 2 adapts dynamically and flows down to origin
    idx = max(2, int(38 + alpha * 32))
    tube2 = build_tube(traj2_x[:idx], traj2_y[:idx], current_v)
    plotter.add_mesh(tube2, color="#0284c7", name="traj2")
    
    step_cam(0.08)
    plotter.write_frame()

for i in range(len(ce_step2)):
    plotter.remove_actor(f"ce2_{i}")

# Trajectory 2 is removed after optimization
plotter.remove_actor("traj2")

for _ in range(8):
    step_cam(0.08)
    plotter.write_frame()

# 3. Learner reshapes basin: k_quad increases from 0.155 to 0.280
# Boundary values of the function rise further, and rho = min_{d D} V(x) grows smoothly!
n_grow2 = 45
k_start2, k_end2 = 0.155, 0.280
for step in range(n_grow2):
    k_quad = k_start2 + (k_end2 - k_start2) * (step + 1) / n_grow2
    update_surface()
    update_sublevel()
    step_cam(0.08)
    plotter.write_frame()

# =============================================================================
# ITERATION 3: New Counterexamples in expanded sublevel set -> Resolve & Remove
# =============================================================================
# Trajectory 3 approaches outer zone defect
for f in range(20):
    idx = max(2, int((f + 1) / 20 * 42))
    tube3 = build_tube(traj3_x[:idx], traj3_y[:idx], current_v)
    plotter.add_mesh(tube3, color="#0284c7", name="traj3")
    step_cam(0.08)
    plotter.write_frame()

# 1. New counterexamples appear in outer band
for s in np.linspace(0.02, 0.075, 10):
    for i, (cx, cy) in enumerate(ce_step3):
        cz = float(current_v(np.array([cx]), np.array([cy]))[0])
        sp = pv.Sphere(radius=s, center=(cx, cy, cz))
        plotter.add_mesh(sp, color="#dc2626", name=f"ce3_{i}")
    step_cam(0.08)
    plotter.write_frame()

for _ in range(15):
    step_cam(0.08)
    plotter.write_frame()

# 2. Inductive synthesis: w3 morphs to 0 (final surface becomes fully convex)
n_m3 = 30
for step in range(n_m3):
    alpha = (step + 1) / n_m3
    w3 = 1.0 - alpha
    wr = max(0.34 * (1.0 - alpha), 0.0)
    update_surface()
    update_sublevel()
    
    # Cex 3 dissolve
    scale = max(0.075 * (1.0 - alpha), 0.001)
    for i, (cx, cy) in enumerate(ce_step3):
        if alpha < 0.95:
            cz = float(current_v(np.array([cx]), np.array([cy]))[0])
            sp = pv.Sphere(radius=scale, center=(cx, cy, cz))
            plotter.add_mesh(sp, color="#dc2626", name=f"ce3_{i}")
        else:
            plotter.remove_actor(f"ce3_{i}")
            
    # Trajectory 3 adapts dynamically and flows down to origin
    idx = max(2, int(42 + alpha * 33))
    tube3 = build_tube(traj3_x[:idx], traj3_y[:idx], current_v)
    plotter.add_mesh(tube3, color="#0284c7", name="traj3")
    
    step_cam(0.08)
    plotter.write_frame()

for i in range(len(ce_step3)):
    plotter.remove_actor(f"ce3_{i}")

# Trajectory 3 is removed after optimization
plotter.remove_actor("traj3")

for _ in range(12):
    step_cam(0.08)
    plotter.write_frame()

# =============================================================================
# FINAL STATE: Strictly Decreasing Lyapunov Function & Trajectory Family
# =============================================================================
traj_colors = ["#0284c7", "#0369a1", "#0284c7", "#0369a1", "#0284c7", "#0369a1", "#0284c7", "#0369a1"]
n_final = 55

def v_final(x, y):
    return 0.28 * (x**2 + y**2) + 0.015 * (x**4 + y**4) / 4.0

for f in range(n_final):
    frac = min((f + 1) / 35.0, 1.0)
    for i, (tx, ty) in enumerate(final_trajs_xy):
        curr_len = max(2, int(frac * len(tx)))
        tube_f = build_tube(tx[:curr_len], ty[:curr_len], v_final, z_offset=0.025, radius=0.016)
        plotter.add_mesh(tube_f, color=traj_colors[i], name=f"final_traj_{i}")
        
    step_cam(0.10)
    plotter.write_frame()

plotter.close()
print("Animation completed successfully: lyapunov_learning_cegis.mp4")