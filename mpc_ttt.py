"""
MPC Steering Control — Winding Corridor
========================================
Cost (exactly as in the formula):

    J = Σ_{i=1}^{N} q_τ (τ_{L,k+i|k} − τ_{R,k+i|k})²  +  Σ_{i=0}^{N-1} r_u u²_{k+i}

Prediction: each KF state propagated with constant-velocity model;
            control u shifts the τ̇ of each side symmetrically.

Constraint: |u_{k+i}| ≤ U_max   (box only — no rate constraint)

Apply:      u_k = u*_{k|k}   (receding horizon, first element only)

Light mode, scene-only layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.optimize  import minimize

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

# ── Physical / corridor parameters ────────────────────────────────────────────
v              = 2.0
dt_phys        = 0.05
TAU_INTERVAL   = 5
N_OBS          = 60
noise_std      = 0.05
cam_half_angle = 0.25
corridor_width = 3.0

# ── MPC parameters ────────────────────────────────────────────────────────────
N_hor  = 10       # prediction horizon N
q_tau  = 1.0      # weight on (τ^L_{k+i|k} − τ^R_{k+i|k})²
r_u    = 25.0     # weight on u²_{k+i}  (high relative to q_tau keeps u sub-saturation)
U_MAX  = 0.3      # |u_{k+i}| ≤ U_max  (only constraint)

# ── Prediction model ──────────────────────────────────────────────────────────
# KF state per side: [τ, τ̇].
# MPC dt matches physics dt so one horizon step = one integration step.
# Control u biases τ̇ symmetrically:
#   τ̇^L -= β*u   (turn left → left gap closes)
#   τ̇^R += β*u   (turn left → right gap opens)
# With dt_kf = dt_phys and small β, the unconstrained optimal u stays
# well inside U_MAX, preventing bang-bang saturation.
dt_kf = dt_phys   # MPC step = physics step = 0.05 s
beta  = 0.40

F_step = np.array([[1.0, dt_kf],
                   [0.0, 1.0]])

# ── MPC solver ────────────────────────────────────────────────────────────────
def solve_mpc(zL: np.ndarray, zR: np.ndarray) -> float:
    """
    Solve: min  Σ_{i=1}^{N} q_τ(τ^L_i − τ^R_i)²  +  Σ_{i=0}^{N-1} r_u u²_i
    s.t.   |u_i| ≤ U_MAX
    Returns u*_{k|k}.
    """
    def cost(U):
        sL = zL.copy()
        sR = zR.copy()
        J  = 0.0
        for i in range(N_hor):
            sL[1] -= beta * U[i]   # control enters τ̇ before stepping
            sR[1] += beta * U[i]
            sL = F_step @ sL
            sR = F_step @ sR
            # Stage cost i=1..N (predicted output error + effort)
            J += q_tau * (sL[0] - sR[0])**2 + r_u * U[i]**2
        return J

    bounds = [(-U_MAX, U_MAX)] * N_hor
    res = minimize(cost, np.zeros(N_hor), method='SLSQP', bounds=bounds,
                   options={'ftol': 1e-9, 'maxiter': 300})
    return float(res.x[0])

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

# ── Winding corridor ──────────────────────────────────────────────────────────
n_pts = 600
xs_c  = np.linspace(0, 80, n_pts)
ys_c  = (4.5 * np.sin(xs_c / 10.0)
       + 2.5 * np.sin(xs_c /  5.5 + 1.1)
       + 1.2 * np.sin(xs_c /  3.0 + 2.3))

raw_dx = np.gradient(xs_c);  raw_dy = np.gradient(ys_c)
mag    = np.sqrt(raw_dx**2 + raw_dy**2)
tx = raw_dx / mag;  ty = raw_dy / mag
nx = -ty;           ny =  tx

half_w = corridor_width / 2.0
lx = xs_c + half_w * nx;  ly = ys_c + half_w * ny
rx = xs_c - half_w * nx;  ry = ys_c - half_w * ny

wall_segs = []
for i in range(n_pts - 1):
    wall_segs.append((lx[i], ly[i], lx[i+1], ly[i+1]))
    wall_segs.append((rx[i], ry[i], rx[i+1], ry[i+1]))

def ray_wall_hit(rx_r, ry_r, th):
    cth, sth = np.cos(th), np.sin(th)
    min_t = np.inf;  hx = hy = None
    for (x1, y1, x2, y2) in wall_segs:
        dx, dy = x2 - x1, y2 - y1
        denom  = cth * dy - sth * dx
        if abs(denom) < 1e-10: continue
        t  = ((x1 - rx_r) * dy  - (y1 - ry_r) * dx) / denom
        s_ = ((x1 - rx_r) * sth - (y1 - ry_r) * cth) / denom
        if t > 0.05 and 0.0 <= s_ <= 1.0 and t < min_t:
            min_t = t;  hx = rx_r + t*cth;  hy = ry_r + t*sth
    return None if hx is None else (hx, hy, min_t / v)

# ── Robot start ───────────────────────────────────────────────────────────────
si    = np.argmin(np.abs(xs_c - 2.0))
x_r   = xs_c[si];  y_r = ys_c[si]
theta = np.arctan2(ty[si], tx[si])

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

        # KF updates
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

        # MPC solve
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
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_MAIN)

ax = fig.add_axes([0.06, 0.07, 0.91, 0.85])
ax.set_facecolor(BG_MAIN)

pad  = 2.5
x_lo = min(path_x) - pad;  x_hi = max(path_x) + pad
y_lo = min(min(path_y), np.min(ly), np.min(ry)) - pad
y_hi = max(max(path_y), np.max(ly), np.max(ry)) + pad

ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=CLR_TEXT, fontsize=13)
ax.set_ylabel("y (m)", color=CLR_TEXT, fontsize=13)
ax.tick_params(colors=CLR_TEXT, labelsize=11)
for sp in ax.spines.values(): sp.set_edgecolor("#aaaaaa")
ax.grid(True, alpha=0.35, color=CLR_GRID)

fill_x = np.concatenate([lx, rx[::-1]])
fill_y = np.concatenate([ly, ry[::-1]])
ax.fill(fill_x, fill_y, color=CLR_FILL, alpha=0.55, zorder=2)
ax.plot(xs_c, ys_c, color=CLR_CENTRE, lw=1.0, ls="--", alpha=0.45, zorder=3)
for wax, way in [(lx, ly), (rx, ry)]:
    ax.plot(wax, way, color=CLR_WALL, lw=2.5, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground=CLR_WALL_GLOW),
                          pe.Normal()])

ax.scatter(path_x[0], path_y[0], s=110, color=CLR_ROBOT,
           marker="D", zorder=12, edgecolors=CLR_ROBOT_EDGE, linewidths=1.5)
ax.annotate("START", (path_x[0], path_y[0]), color=CLR_TEXT,
            textcoords="offset points", xytext=(6, 8),
            fontsize=10, fontweight="bold")

# ── Animated artists ──────────────────────────────────────────────────────────
robot_body,  = ax.plot([], [], "s", ms=14, color=CLR_ROBOT,
                        markeredgecolor=CLR_ROBOT_EDGE, markeredgewidth=2, zorder=10)
robot_trail, = ax.plot([], [], "-", color=CLR_ROBOT, lw=1.8, alpha=0.40, zorder=5)
heading_q    = ax.quiver([0], [0], [0], [0],
                          color=CLR_ROBOT, scale=18, scale_units="xy",
                          angles="xy", width=0.012, headwidth=4,
                          headlength=5, zorder=11)

cam_ray_L, = ax.plot([], [], "--", color=CLR_LEFT,    lw=2, alpha=0.85, zorder=7)
feat_L,    = ax.plot([], [], "o",  ms=12, color=CLR_LEFT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=11)
trail_L,   = ax.plot([], [], "o",  ms=4, color=CLR_LEFT_TR, alpha=0.40, zorder=6)

cam_ray_R, = ax.plot([], [], "--", color=CLR_RIGHT,   lw=2, alpha=0.85, zorder=7)
feat_R,    = ax.plot([], [], "o",  ms=12, color=CLR_RIGHT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=11)
trail_R,   = ax.plot([], [], "o",  ms=4, color=CLR_RIGHT_TR, alpha=0.40, zorder=6)

seg_line, = ax.plot([], [], "-",  color=CLR_SEG,      lw=2.5, alpha=0.9,  zorder=9)
past_segs = [ax.plot([], [], "-", color=CLR_SEG_PAST, lw=0.8,
                     alpha=0.20, zorder=4)[0] for _ in range(N_OBS)]

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
    Line2D([0],[0], color=CLR_WALL,  lw=2.5, label="Corridor walls"),
    Line2D([0],[0], color=CLR_LEFT,  lw=2, ls="--",
           label=r"Left ray → $O^L_k$"),
    Line2D([0],[0], color=CLR_RIGHT, lw=2, ls="--",
           label=r"Right ray → $O^R_k$"),
    Line2D([0],[0], color=CLR_SEG,   lw=2.5,
           label=r"Segment $O^L_k$ → $O^R_k$"),
]
ax.legend(handles=legend_els, loc="lower right",
          facecolor=BG_HUD, labelcolor=CLR_TEXT,
          fontsize=11, framealpha=0.95, edgecolor="#aaaaaa")

ax.set_title(
    rf"MPC Corridor Centering   $u_k = u^*_{{k|k}}$,  "
    rf"$N={N_hor}$,  $q_{{\tau}}={q_tau}$,  $r_u={r_u}$,  $|u|\leq{U_MAX}$",
    color=CLR_TEXT, fontsize=12, fontweight="bold", pad=6)

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
        f"LEFT ray\n"
        f"  τ obs    : {tau_left_obs_all[k]:.4f} s\n"
        f"  τ̂_{{k|k}} : {tau_left_est_all[k]:.4f} s   σ={sigma_left_all[k]:.4f}")

    obs_r_t.set_text(
        f"RIGHT ray\n"
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
