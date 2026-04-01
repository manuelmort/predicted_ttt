"""
MPC Steering Control — Stadium (Oval) Track
=============================================
Cost:
    J = Σ_{i=1}^{N} q_τ (τ_{L,k+i|k} − τ_{R,k+i|k})²  +  Σ_{i=0}^{N-1} r_u u²_{k+i}

Constraint:  |u_{k+i}| ≤ U_max   (box only)
Apply:       u_k = u*_{k|k}   (receding horizon)

Track: two straight sections joined by two semicircular turns — a stadium
       (oval) shape.  Robot drives CCW.  Camera is front-facing (±0.25 rad),
       giving symmetric τ^L ≈ τ^R on the straights and interesting dynamics
       through the turns.
Light mode, scene-only layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.lines  import Line2D
from scipy.optimize    import minimize
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_MAIN        = "#ffffff"
BG_HUD         = "#f5f5f5"
CLR_TEXT       = "#111111"
CLR_GRID       = "#cccccc"
CLR_WALL       = "#1a5fa0"
CLR_WALL_GLOW  = "#5599cc"
CLR_FILL       = "#ddeeff"
CLR_CENTRE     = "#7aaadd"
CLR_ROBOT      = "#55c47a"
CLR_ROBOT_EDGE = "#1a6b3a"
CLR_LEFT       = "#c85000"
CLR_LEFT_TR    = "#e07030"
CLR_RIGHT      = "#007b8a"
CLR_RIGHT_TR   = "#33aabb"
CLR_SEG        = "#333333"
CLR_SEG_PAST   = "#aaaaaa"

# ── Physical parameters ───────────────────────────────────────────────────────
v              = 2.0
dt_phys        = 0.05
TAU_INTERVAL   = 5
N_OBS          = 80
noise_std      = 0.05
cam_half_angle = 0.25       # front-facing ±0.25 rad (~14°)
corridor_width = 6.0

# ── MPC parameters ────────────────────────────────────────────────────────────
N_hor  = 10
q_tau  = 1.0
r_u    = 25.0
U_MAX  = 0.3

# ── Prediction model ──────────────────────────────────────────────────────────
dt_kf  = dt_phys
beta   = 0.40
F_step = np.array([[1.0, dt_kf],
                   [0.0, 1.0]])

# ── Stadium track geometry ────────────────────────────────────────────────────
R_turn     = 15.0     # semicircle radius — keep gentle
L_straight = 25.0     # length of each straight section
half_w     = corridor_width / 2.0

n_straight = 200
n_turn     = 120

# CCW: start bottom-right, go LEFT along bottom, turn LEFT at left end,
#      go RIGHT along top, turn RIGHT at right end
# Bottom straight  (right → left, y = -R_turn)
xs1 = np.linspace( L_straight/2, -L_straight/2, n_straight)
ys1 = np.full(n_straight, -R_turn)

# Left semicircle  (bottom-left to top-left, centre at (-L/2, 0))
a_l = np.linspace(-np.pi/2, np.pi/2, n_turn)
xs2 = -L_straight/2 + R_turn * np.cos(a_l + np.pi)
ys2 = R_turn * np.sin(a_l)

# Top straight     (left → right, y = +R_turn)
xs3 = np.linspace(-L_straight/2,  L_straight/2, n_straight)
ys3 = np.full(n_straight,  R_turn)

# Right semicircle (top-right to bottom-right, centre at (+L/2, 0))
a_r = np.linspace(np.pi/2, -np.pi/2, n_turn)
xs4 = L_straight/2 + R_turn * np.cos(a_r)
ys4 = R_turn * np.sin(a_r)

xs_c = np.concatenate([xs1, xs2, xs3, xs4])
ys_c = np.concatenate([ys1, ys2, ys3, ys4])
n_pts = len(xs_c)

# Tangent and left-pointing normal
dx = np.gradient(xs_c);  dy = np.gradient(ys_c)
mag = np.sqrt(dx**2 + dy**2)
tx_arr = dx / mag;  ty_arr = dy / mag
nx_arr = -ty_arr;   ny_arr =  tx_arr   # rotated 90° CCW = left side

lx = xs_c + half_w * nx_arr;  ly = ys_c + half_w * ny_arr   # left wall
rx = xs_c - half_w * nx_arr;  ry = ys_c - half_w * ny_arr   # right wall

# Closed wall segments
wall_segs = []
for i in range(n_pts):
    j = (i + 1) % n_pts
    wall_segs.append((lx[i], ly[i], lx[j], ly[j]))
    wall_segs.append((rx[i], ry[i], rx[j], ry[j]))

def ray_wall_hit(rx_r, ry_r, th):
    cth, sth = np.cos(th), np.sin(th)
    min_t = np.inf;  hx = hy = None
    for (x1, y1, x2, y2) in wall_segs:
        dx2, dy2 = x2 - x1, y2 - y1
        denom    = cth * dy2 - sth * dx2
        if abs(denom) < 1e-10: continue
        t  = ((x1 - rx_r) * dy2  - (y1 - ry_r) * dx2) / denom
        s_ = ((x1 - rx_r) * sth  - (y1 - ry_r) * cth) / denom
        if t > 0.05 and 0.0 <= s_ <= 1.0 and t < min_t:
            min_t = t;  hx = rx_r + t*cth;  hy = ry_r + t*sth
    return None if hx is None else (hx, hy, min_t / v)

# ── Kalman filter ─────────────────────────────────────────────────────────────
F_kf = F_step.copy()
H_kf = np.array([[1.0, 0.0]])
Q_kf = np.diag([0.01, 0.005])
R_kf = np.array([[noise_std**2]])

def kf_predict(z, P):
    return F_kf @ z, F_kf @ P @ F_kf.T + Q_kf

def kf_update(z, P, y):
    inn = y - (H_kf @ z).item()
    S   = H_kf @ P @ H_kf.T + R_kf
    K   = P @ H_kf.T @ np.linalg.inv(S)
    return z + (K * inn).flatten(), (np.eye(2) - K @ H_kf) @ P

# ── MPC solver ────────────────────────────────────────────────────────────────
def solve_mpc(zL: np.ndarray, zR: np.ndarray) -> float:
    def cost(U):
        sL = zL.copy();  sR = zR.copy();  J = 0.0
        for i in range(N_hor):
            sL[1] -= beta * U[i]
            sR[1] += beta * U[i]
            sL = F_step @ sL;  sR = F_step @ sR
            J += q_tau * (sL[0] - sR[0])**2 + r_u * U[i]**2
        return J
    res = minimize(cost, np.zeros(N_hor), method='SLSQP',
                   bounds=[(-U_MAX, U_MAX)] * N_hor,
                   options={'ftol': 1e-9, 'maxiter': 300})
    return float(res.x[0])

# ── Robot start: bottom-right corner, heading CCW (left = π) ─────────────────
x_r   =  L_straight / 2
y_r   = -R_turn
theta =  np.pi          # pointing left

path_x, path_y, path_theta = [x_r], [y_r], [theta]
obs_frames        = []
tau_left_obs_all  = [];  tau_left_est_all  = [];  sigma_left_all  = []
tau_right_obs_all = [];  tau_right_est_all = [];  sigma_right_all = []
wall_left_all     = [];  wall_right_all    = []
ctrl_all          = [];  diff_tau_all      = []

z_kf_L = None;  P_kf_L = None
z_kf_R = None;  P_kf_R = None
current_u = 0.0
frame_idx = 0;  obs_count = 0

print("Pre-simulating…")
while obs_count < N_OBS:
    if frame_idx % TAU_INTERVAL == 0:
        hit_left  = ray_wall_hit(x_r, y_r, theta + cam_half_angle)
        hit_right = ray_wall_hit(x_r, y_r, theta - cam_half_angle)
        if hit_left is None or hit_right is None:
            print(f"Ray miss at obs {obs_count}"); break

        xwl, ywl, tau_L_true = hit_left
        xwr, ywr, tau_R_true = hit_right
        tau_L_noisy = tau_L_true + np.random.normal(0, noise_std)
        tau_R_noisy = tau_R_true + np.random.normal(0, noise_std)

        if z_kf_L is None:
            z_kf_L = np.array([tau_L_noisy, 0.0])
            P_kf_L = np.diag([noise_std**2, 0.5])
        else:
            z_kf_L, P_kf_L = kf_predict(z_kf_L, P_kf_L)
            z_kf_L, P_kf_L = kf_update(z_kf_L, P_kf_L, tau_L_noisy)

        if z_kf_R is None:
            z_kf_R = np.array([tau_R_noisy, 0.0])
            P_kf_R = np.diag([noise_std**2, 0.5])
        else:
            z_kf_R, P_kf_R = kf_predict(z_kf_R, P_kf_R)
            z_kf_R, P_kf_R = kf_update(z_kf_R, P_kf_R, tau_R_noisy)

        u = 0.0 if obs_count == 0 else solve_mpc(z_kf_L.copy(), z_kf_R.copy())
        current_u = u

        obs_frames.append(frame_idx)
        tau_left_obs_all.append(tau_L_noisy)
        tau_left_est_all.append(z_kf_L[0])
        sigma_left_all.append(np.sqrt(P_kf_L[0, 0]))
        wall_left_all.append((xwl, ywl))

        tau_right_obs_all.append(tau_R_noisy)
        tau_right_est_all.append(z_kf_R[0])
        sigma_right_all.append(np.sqrt(P_kf_R[0, 0]))
        wall_right_all.append((xwr, ywr))

        ctrl_all.append((u, z_kf_L[0], z_kf_R[0]))
        diff_tau_all.append(z_kf_L[0] - z_kf_R[0])
        obs_count += 1

    theta += current_u * dt_phys
    x_r   += v * np.cos(theta) * dt_phys
    y_r   += v * np.sin(theta) * dt_phys
    path_x.append(x_r);  path_y.append(y_r);  path_theta.append(theta)
    frame_idx += 1

N_phys       = len(path_x)
TOTAL_FRAMES = N_phys + 60
print(f"Done — {obs_count} obs, {N_phys} frames.")

# ── Figure ─────────────────────────────────────────────────────────────────────
x_extent = L_straight/2 + R_turn + half_w + 4.0
y_extent = R_turn + half_w + 4.0
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(BG_MAIN)

ax = fig.add_axes([0.04, 0.05, 0.93, 0.90])
ax.set_facecolor(BG_MAIN)
ax.set_xlim(-x_extent, x_extent)
ax.set_ylim(-y_extent, y_extent)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=CLR_TEXT, fontsize=13)
ax.set_ylabel("y (m)", color=CLR_TEXT, fontsize=13)
ax.tick_params(colors=CLR_TEXT, labelsize=11)
for sp in ax.spines.values(): sp.set_edgecolor("#aaaaaa")
ax.grid(True, alpha=0.30, color=CLR_GRID)

# ── Static track ──────────────────────────────────────────────────────────────
# Filled corridor (left wall polygon − right wall polygon)
fill_x = np.concatenate([lx, rx[::-1]])
fill_y = np.concatenate([ly, ry[::-1]])
ax.fill(fill_x, fill_y, color=CLR_FILL, alpha=0.60, zorder=2)

# Centreline
ax.plot(xs_c, ys_c, color=CLR_CENTRE, lw=1.0, ls="--", alpha=0.45, zorder=3)

# Walls
for wx, wy in [(lx, ly), (rx, ry)]:
    ax.plot(wx, wy, color=CLR_WALL, lw=2.5, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground=CLR_WALL_GLOW),
                          pe.Normal()])

# CCW direction arrows on centreline
for frac in [0.08, 0.33, 0.58, 0.83]:
    idx = int(frac * n_pts)
    ax.annotate("", xy=(xs_c[idx]+tx_arr[idx]*2, ys_c[idx]+ty_arr[idx]*2),
                xytext=(xs_c[idx], ys_c[idx]),
                arrowprops=dict(arrowstyle="-|>", color=CLR_CENTRE,
                                lw=1.4, mutation_scale=14),
                zorder=5)

# Start marker
ax.scatter(path_x[0], path_y[0], s=120, color=CLR_ROBOT,
           marker="D", zorder=14, edgecolors=CLR_ROBOT_EDGE, linewidths=1.5)
ax.annotate("START", (path_x[0], path_y[0]), color=CLR_TEXT,
            textcoords="offset points", xytext=(8, 10),
            fontsize=10, fontweight="bold")

# ── Animated artists ──────────────────────────────────────────────────────────
robot_body,  = ax.plot([], [], "s", ms=14, color=CLR_ROBOT,
                        markeredgecolor=CLR_ROBOT_EDGE, markeredgewidth=2, zorder=12)
robot_trail, = ax.plot([], [], "-", color=CLR_ROBOT, lw=1.8, alpha=0.40, zorder=6)
heading_q    = ax.quiver([0], [0], [0], [0],
                          color=CLR_ROBOT, scale=15, scale_units="xy",
                          angles="xy", width=0.008, headwidth=4,
                          headlength=5, zorder=13)

cam_ray_L, = ax.plot([], [], "--", color=CLR_LEFT,    lw=2,  alpha=0.85, zorder=8)
feat_L,    = ax.plot([], [], "o",  ms=12, color=CLR_LEFT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=13)
trail_L,   = ax.plot([], [], "o",  ms=4,  color=CLR_LEFT_TR, alpha=0.40, zorder=7)

cam_ray_R, = ax.plot([], [], "--", color=CLR_RIGHT,   lw=2,  alpha=0.85, zorder=8)
feat_R,    = ax.plot([], [], "o",  ms=12, color=CLR_RIGHT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=13)
trail_R,   = ax.plot([], [], "o",  ms=4,  color=CLR_RIGHT_TR, alpha=0.40, zorder=7)

seg_line, = ax.plot([], [], "-",  color=CLR_SEG,      lw=2.5, alpha=0.9,  zorder=11)
past_segs = [ax.plot([], [], "-", color=CLR_SEG_PAST, lw=0.8,
                     alpha=0.20, zorder=5)[0] for _ in range(N_OBS)]

# ── HUD ───────────────────────────────────────────────────────────────────────
hud_kw = dict(transform=ax.transAxes, va="top", fontfamily="monospace",
              fontsize=10, bbox=dict(boxstyle="round,pad=0.4",
                                     facecolor=BG_HUD, edgecolor="#bbbbbb",
                                     alpha=0.88))
title_t = ax.text(0.01, 0.99, "", color=CLR_TEXT,  **hud_kw)
obs_t   = ax.text(0.01, 0.91, "", color=CLR_LEFT,  **hud_kw)
obs_r_t = ax.text(0.01, 0.78, "", color=CLR_RIGHT, **hud_kw)
mpc_t   = ax.text(0.01, 0.65, "", color="#444444", **hud_kw)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_els = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=CLR_ROBOT,
           markersize=11, label="Robot", markeredgecolor=CLR_ROBOT_EDGE),
    Line2D([0],[0], color=CLR_WALL,  lw=2.5, label="Track walls"),
    Line2D([0],[0], color=CLR_LEFT,  lw=2, ls="--",
           label=r"Left ray → $O^L_k$   (+0.25 rad)"),
    Line2D([0],[0], color=CLR_RIGHT, lw=2, ls="--",
           label=r"Right ray → $O^R_k$  (−0.25 rad)"),
    Line2D([0],[0], color=CLR_SEG,   lw=2.5,
           label=r"Segment $O^L_k$ → $O^R_k$"),
]
ax.legend(handles=legend_els, loc="lower right",
          facecolor=BG_HUD, labelcolor=CLR_TEXT,
          fontsize=11, framealpha=0.95, edgecolor="#aaaaaa")

ax.set_title(
    rf"MPC Oval Track Centering   $u_k = u^*_{{k|k}}$,  "
    rf"$N={N_hor}$,  $q_{{\tau}}={q_tau}$,  $r_u={r_u}$,  $|u|\leq{U_MAX}$,  "
    rf"cam $\pm${np.degrees(cam_half_angle):.0f}°",
    color=CLR_TEXT, fontsize=12, fontweight="bold", pad=8)

# ── Animation ─────────────────────────────────────────────────────────────────
def get_k(frame):
    k = -1
    for i, f in enumerate(obs_frames):
        if frame >= f: k = i
    return k

def animate(frame):
    f  = min(frame, N_phys - 1)
    xr, yr, th = path_x[f], path_y[f], path_theta[f]

    robot_body.set_data([xr], [yr])
    heading_q.set_offsets([[xr, yr]])
    heading_q.set_UVC(np.cos(th), np.sin(th))
    robot_trail.set_data(path_x[:f+1], path_y[:f+1])

    k = get_k(frame)
    if k < 0:
        for art in [cam_ray_L, feat_L, trail_L,
                    cam_ray_R, feat_R, trail_R, seg_line]:
            art.set_data([], [])
        title_t.set_text("Initialising…")
        return []

    xwl, ywl = wall_left_all[k]
    cam_ray_L.set_data([xr, xwl], [yr, ywl])
    feat_L.set_data([xwl], [ywl])
    trail_L.set_data([wall_left_all[i][0] for i in range(k+1)],
                     [wall_left_all[i][1] for i in range(k+1)])

    xwr, ywr = wall_right_all[k]
    cam_ray_R.set_data([xr, xwr], [yr, ywr])
    feat_R.set_data([xwr], [ywr])
    trail_R.set_data([wall_right_all[i][0] for i in range(k+1)],
                     [wall_right_all[i][1] for i in range(k+1)])

    seg_line.set_data([xwl, xwr], [ywl, ywr])
    for i, seg in enumerate(past_segs):
        if 0 < i <= k:
            x0, y0 = wall_left_all[i-1]
            x1, y1 = wall_right_all[i-1]
            seg.set_data([x0, x1], [y0, y1])
        else:
            seg.set_data([], [])

    u_k, tau_lp, tau_rp = ctrl_all[k]
    dtau = diff_tau_all[k]

    title_t.set_text(
        f"k = {k:3d}   frame {f}/{N_phys}"
        + ("   ◀ DONE ▶" if frame >= N_phys else ""))

    obs_t.set_text(
        f"LEFT ray  (+{np.degrees(cam_half_angle):.0f}°)\n"
        f"  τ obs    : {tau_left_obs_all[k]:.4f} s\n"
        f"  τ̂_{{k|k}} : {tau_left_est_all[k]:.4f} s   σ={sigma_left_all[k]:.4f}")

    obs_r_t.set_text(
        f"RIGHT ray (−{np.degrees(cam_half_angle):.0f}°)\n"
        f"  τ obs    : {tau_right_obs_all[k]:.4f} s\n"
        f"  τ̂_{{k|k}} : {tau_right_est_all[k]:.4f} s   σ={sigma_right_all[k]:.4f}")

    mpc_t.set_text(
        f"MPC  N={N_hor},  |u|≤{U_MAX}\n"
        f"  Δτ = τ̂^L−τ̂^R = {dtau:+.4f} s\n"
        f"  u_k = u*_{{k|k}} = {u_k:+.4f} rad\n"
        f"  θ = {np.degrees(th):+.2f}°")

    return (robot_body, robot_trail, heading_q,
            cam_ray_L, feat_L, trail_L,
            cam_ray_R, feat_R, trail_R,
            seg_line, title_t, obs_t, obs_r_t, mpc_t,
            *past_segs)

ani = animation.FuncAnimation(
    fig, animate,
    frames=TOTAL_FRAMES,
    interval=30,
    blit=False,
    repeat=True
)

plt.tight_layout(pad=0)
plt.show()
