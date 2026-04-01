"""
Kalman Filter + Steering Control — Circular Obstacle
Animation only — trajectory scene + HUD (no inset plot)
Light mode, suitable for recording.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

np.random.seed(7)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_MAIN       = "#ffffff"
BG_HUD        = "#f5f5f5"
CLR_TEXT      = "#111111"
CLR_GRID      = "#cccccc"
CLR_ROAD      = "#e0e0e0"
CLR_ROAD_LINE = "#aaaaaa"

CLR_OBS_FILL  = "#ddeeff"
CLR_OBS_RING  = "#1a5fa0"
CLR_OBS_GLOW  = "#5599cc"

CLR_ROBOT     = "#55c47a"
CLR_ROBOT_EDGE= "#1a6b3a"

CLR_LEFT_RAY  = "#c85000"
CLR_LEFT_WALL = "#e07030"

CLR_RIGHT_RAY = "#007b8a"
CLR_RIGHT_WALL= "#33aabb"

CLR_SEG       = "#333333"
CLR_SEG_PAST  = "#888888"

# ── Tuneable ──────────────────────────────────────────────────────────────────
v              = 0.7
dt_phys        = 0.05
TAU_INTERVAL   = 20
N_OBS          = 72
K_ctrl         = 5.5
U_MAX          = 0.12
noise_std      = 0.05
cam_half_angle = 0.08

# ── Circular obstacle ─────────────────────────────────────────────────────────
obs_x, obs_y, obs_r = 3.0, 7.5, 5.5

def ray_circle_hit(rx, ry, th_cam):
    cx, cy = np.cos(th_cam), np.sin(th_cam)
    dx, dy = rx - obs_x, ry - obs_y
    b    = dx*cx + dy*cy
    c    = dx**2 + dy**2 - obs_r**2
    disc = b**2 - c
    if disc < 0:
        return None
    for t in [-b - np.sqrt(disc), -b + np.sqrt(disc)]:
        if t > 0.05:
            return rx + t*cx, ry + t*cy, t / v
    return None

# ── Kalman helpers ────────────────────────────────────────────────────────────
dt_kf = 1.0
F_kf  = np.array([[1, dt_kf], [0, 1]])
H_kf  = np.array([[1, 0]])
Q_kf  = np.diag([0.01, 0.005])
R_kf  = np.array([[noise_std**2]])

def kf_predict(z, P):
    return F_kf @ z, F_kf @ P @ F_kf.T + Q_kf

def kf_update(z, P, y):
    inn = y - (H_kf @ z).item()
    S   = H_kf @ P @ H_kf.T + R_kf
    K   = P @ H_kf.T @ np.linalg.inv(S)
    return z + (K * inn).flatten(), (np.eye(2) - K @ H_kf) @ P

# ── Pre-simulate ──────────────────────────────────────────────────────────────
x_r, y_r, theta = 0.5, 0.0, np.radians(1)

path_x, path_y, path_theta = [x_r], [y_r], [theta]
obs_frames     = []
tau_obs_all    = []
tau_est_all    = []
tau_pred_all   = []
sigma_all      = []
ctrl_all       = []
wall_left_all  = []
wall_right_all = []

z_kf = None; P_kf = None
current_u = 0.0
frame_idx = 0; obs_count = 0

while obs_count < N_OBS:
    if frame_idx % TAU_INTERVAL == 0:
        th_base  = theta + np.pi / 2
        th_left  = th_base + cam_half_angle
        hit_left = ray_circle_hit(x_r, y_r, th_left)

        if hit_left is None:
            print(f"Ray miss at obs {obs_count}"); break

        xwl, ywl, tau_true = hit_left
        tau_noisy = tau_true + np.random.normal(0, noise_std)

        if z_kf is None:
            z_kf = np.array([tau_noisy, 0.0])
            P_kf = np.diag([noise_std**2, 0.5])
        else:
            z_kf, P_kf = kf_predict(z_kf, P_kf)
            z_kf, P_kf = kf_update(z_kf, P_kf, tau_noisy)

        z_ahead, _ = kf_predict(z_kf, P_kf)
        tau_ahead  = z_ahead[0]

        th_right = th_base - cam_half_angle
        xwr = x_r + tau_ahead * v * np.cos(th_right)
        ywr = y_r + tau_ahead * v * np.sin(th_right)

        u = 0.0 if obs_count == 0 else \
            np.clip(K_ctrl * (tau_ahead - z_kf[0]), -U_MAX, U_MAX)
        current_u = u

        obs_frames.append(frame_idx)
        tau_obs_all.append(tau_noisy)
        tau_est_all.append(z_kf[0])
        tau_pred_all.append(tau_ahead)
        sigma_all.append(np.sqrt(P_kf[0, 0]))
        ctrl_all.append((u, tau_ahead, z_kf[0]))
        wall_left_all.append((xwl, ywl))
        wall_right_all.append((xwr, ywr))
        obs_count += 1

    theta += current_u * dt_phys
    x_r   += v * np.cos(theta) * dt_phys
    y_r   += v * np.sin(theta) * dt_phys

    path_x.append(x_r); path_y.append(y_r); path_theta.append(theta)
    frame_idx += 1

N_phys       = len(path_x)
TOTAL_FRAMES = N_phys + 60

# ── Figure — scene only ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 13))
fig.patch.set_facecolor(BG_MAIN)

ax = fig.add_axes([0.09, 0.07, 0.88, 0.83])
ax.set_facecolor(BG_MAIN)

x_lo = min(path_x) - 7.5
x_hi = max(path_x) + 5.0
y_lo = min(min(path_y) - 5.0, -5.5)
y_hi = obs_y + obs_r + 6.0

ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=CLR_TEXT, fontsize=14)
ax.set_ylabel("y (m)", color=CLR_TEXT, fontsize=14)
ax.tick_params(colors=CLR_TEXT, labelsize=12)
for sp in ax.spines.values(): sp.set_edgecolor("#aaaaaa")
ax.grid(True, alpha=0.45, color=CLR_GRID)

# ── Static scene ──────────────────────────────────────────────────────────────
ax.fill_between([x_lo, x_hi], [-0.3]*2, [0.3]*2, color=CLR_ROAD, zorder=1)
ax.axhline(0, color=CLR_ROAD_LINE, lw=0.9, ls="--", zorder=2)

obs_patch = mpatches.Circle((obs_x, obs_y), obs_r, color=CLR_OBS_FILL, zorder=3)
ax.add_patch(obs_patch)
theta_ring = np.linspace(0, 2*np.pi, 400)
ax.plot(obs_x + obs_r*np.cos(theta_ring),
        obs_y + obs_r*np.sin(theta_ring),
        color=CLR_OBS_RING, lw=3.0, zorder=4,
        path_effects=[pe.Stroke(linewidth=5, foreground=CLR_OBS_GLOW),
                      pe.Normal()])
ax.scatter(obs_x, obs_y, s=60, color=CLR_OBS_RING, zorder=5, marker="+")
ax.annotate(f"obstacle\nr={obs_r}m", (obs_x, obs_y), color=CLR_OBS_RING,
            textcoords="offset points", xytext=(8, -18), fontsize=11, ha="left")
ax.scatter(obs_x, obs_y - obs_r, s=100, color=CLR_OBS_RING, zorder=5, marker="v")
ax.annotate("closest\npoint", (obs_x, obs_y - obs_r), color=CLR_OBS_RING,
            textcoords="offset points", xytext=(8, -28), fontsize=10)

# ── Animated artists ──────────────────────────────────────────────────────────
robot_body,  = ax.plot([], [], "s", ms=18, color=CLR_ROBOT,
                        markeredgecolor=CLR_ROBOT_EDGE, markeredgewidth=2, zorder=8)
robot_trail, = ax.plot([], [], "-", color=CLR_ROBOT, lw=1.8, alpha=0.40, zorder=5)
heading_q    = ax.quiver([0], [0], [0], [0],
                          color=CLR_ROBOT, scale=1, scale_units="xy",
                          angles="xy", width=0.006, headwidth=4,
                          headlength=5, zorder=9)

cam_ray_left,    = ax.plot([], [], "--", color=CLR_LEFT_RAY,  lw=2,  alpha=0.85, zorder=6)
feat_left,       = ax.plot([], [], "o",  ms=13, color=CLR_LEFT_RAY,
                             markeredgecolor="#333333", markeredgewidth=1.2, zorder=10)
wall_trail_left, = ax.plot([], [], "o",  ms=5,  color=CLR_LEFT_WALL, alpha=0.45, zorder=7)

pred_ray,   = ax.plot([], [], ":",  color=CLR_RIGHT_RAY,  lw=2,  alpha=0.80, zorder=6)
feat_pred,  = ax.plot([], [], "*",  ms=16, color=CLR_RIGHT_RAY,
                       markeredgecolor="#333333", markeredgewidth=1.0, zorder=10)
pred_trail, = ax.plot([], [], "*",  ms=5,  color=CLR_RIGHT_WALL, alpha=0.35, zorder=7)

seg_line, = ax.plot([], [], "-",  color=CLR_SEG,      lw=2.5, alpha=0.9,  zorder=9)
past_segs = [ax.plot([], [], "-", color=CLR_SEG_PAST, lw=0.8,
                     alpha=0.20, zorder=4)[0] for _ in range(N_OBS)]

# ── HUD — placed inside the axes (top-left corner) ───────────────────────────
hud_kw = dict(transform=ax.transAxes, va="top", fontfamily="monospace",
              fontsize=10.5, bbox=dict(boxstyle="round,pad=0.4",
                                       facecolor=BG_HUD, edgecolor="#bbbbbb",
                                       alpha=0.88))
title_t = ax.text(0.01, 0.99, "",  color=CLR_TEXT,       **hud_kw)
obs_t   = ax.text(0.01, 0.90, "",  color=CLR_LEFT_RAY,   **hud_kw)
kf_t    = ax.text(0.01, 0.72, "",  color="#1a7a40",       **hud_kw)
ctrl_t  = ax.text(0.01, 0.58, "",  color=CLR_RIGHT_RAY,  **hud_kw)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_els = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=CLR_ROBOT,
           markersize=11, label="Robot", markeredgecolor=CLR_ROBOT_EDGE),
    Line2D([0],[0], color=CLR_LEFT_RAY,  lw=2, ls="--",
           label=r"Left ray → $O_k$  (τ_k, KF input)"),
    Line2D([0],[0], color=CLR_RIGHT_RAY, lw=2, ls=":",
           label=r"Right → $O_{k+1}$ predicted"),
    Line2D([0],[0], color=CLR_SEG, lw=2.5,
           label=r"Segment $O_k$ → $O_{k+1}$"),
    Line2D([0],[0], color=CLR_OBS_RING, lw=3,
           label="Circular obstacle"),
]
ax.legend(handles=legend_els, loc="lower right",
          facecolor=BG_HUD, labelcolor=CLR_TEXT,
          fontsize=11, framealpha=0.95, edgecolor="#aaaaaa")

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
    heading_q.set_UVC(1.5*np.cos(th), 1.5*np.sin(th))
    robot_trail.set_data(path_x[:f+1], path_y[:f+1])

    k = get_k(frame)
    if k < 0:
        for art in [cam_ray_left, feat_left, wall_trail_left,
                    pred_ray, feat_pred, pred_trail, seg_line]:
            art.set_data([], [])
        title_t.set_text("Initialising…")
        return []

    # Left ray — O_k
    xwl, ywl = wall_left_all[k]
    cam_ray_left.set_data([xr, xwl], [yr, ywl])
    feat_left.set_data([xwl], [ywl])
    wall_trail_left.set_data([wall_left_all[i][0] for i in range(k+1)],
                              [wall_left_all[i][1] for i in range(k+1)])

    # Right predicted — O_{k+1}
    xwr, ywr = wall_right_all[k]
    pred_ray.set_data([xr, xwr], [yr, ywr])
    feat_pred.set_data([xwr], [ywr])
    pred_trail.set_data([wall_right_all[i][0] for i in range(k+1)],
                         [wall_right_all[i][1] for i in range(k+1)])

    # Segment + history
    seg_line.set_data([xwl, xwr], [ywl, ywr])
    for i, seg in enumerate(past_segs):
        if 0 < i <= k:
            x0, y0 = wall_left_all[i-1]
            x1, y1 = wall_right_all[i-1]
            seg.set_data([x0, x1], [y0, y1])
        else:
            seg.set_data([], [])

    u_k, tau_ah, tau_post = ctrl_all[k]

    title_t.set_text(
        f"k = {k:3d}   frame {f}/{N_phys}"
        + ("   ◀ DONE ▶" if frame >= N_phys else ""))

    obs_t.set_text(
        f"τ_k  observed  : {tau_obs_all[k]:.4f} s\n"
        f"τ̂_{{k|k}}  posterior : {tau_est_all[k]:.4f} s\n"
        f"τ̂_{{k+1|k}} lookahead : {tau_pred_all[k]:.4f} s\n"
        f"σ_k            : {sigma_all[k]:.4f}")

    kf_t.set_text(
        f"θ_robot : {np.degrees(th):+.2f}°\n"
        f"innov   = τ_k − τ̂_{{k|k}} = {tau_obs_all[k]-tau_est_all[k]:+.4f} s")

    ctrl_t.set_text(
        f"u = K·(τ̂_{{k+1|k}} − τ̂_{{k|k}})\n"
        f"  Δτ = {tau_ah - tau_post:+.4f}\n"
        f"  u_k = {u_k:+.4f} rad")

    return (robot_body, robot_trail, heading_q,
            cam_ray_left, feat_left, wall_trail_left,
            pred_ray, feat_pred, pred_trail,
            seg_line, title_t, obs_t, kf_t, ctrl_t,
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
