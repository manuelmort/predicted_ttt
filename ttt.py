"""
Steering Control — NO Kalman Filter — Circular Obstacle
========================================================
Control law:  u_k = K * (τ_k - τ_{k-1})

Raw observed τ values used directly — no smoothing, no prediction.
One-step-ahead approximated by linear extrapolation:
    τ_{k+1} ≈ 2·τ_k - τ_{k-1}  →  u_k = K·(τ_k - τ_{k-1})

Compare with kalman_ttt.py to see effect of KF smoothing.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

np.random.seed(7)

# ── Tuneable (same as KF version for fair comparison) ─────────────────────────
v             = 1.0
dt_phys       = 0.05
TAU_INTERVAL  = 20
N_OBS         = 72
K_ctrl        = 5.5
U_MAX         = 0.12
noise_std     = 0.05

# ── Circular obstacle ─────────────────────────────────────────────────────────
obs_x = 3.0
obs_y = 7.5
obs_r = 5.5

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

# ── Pre-simulate (raw τ, no KF) ───────────────────────────────────────────────
x_r, y_r, theta = 0.5, 0.0, np.radians(1)

path_x, path_y, path_theta = [x_r], [y_r], [theta]
obs_frames   = []
tau_obs_all  = []   # raw noisy observations
tau_pred_all = []   # linear extrapolation: 2*τ_k - τ_{k-1}
ctrl_all     = []
wall_hit_all = []

current_u = 0.0
frame_idx = 0; obs_count = 0

while obs_count < N_OBS:
    if frame_idx % TAU_INTERVAL == 0:
        th_cam = theta + np.pi / 2
        hit = ray_circle_hit(x_r, y_r, th_cam)
        if hit is None:
            print(f"Ray miss at obs {obs_count}"); break
        xw, yw, tau_true = hit
        tau_noisy = tau_true + np.random.normal(0, noise_std)

        # No KF — use raw values directly
        # Extrapolate: τ_{k+1} ≈ 2*τ_k - τ_{k-1}
        if obs_count == 0:
            tau_ahead = tau_noisy   # no history yet
            u = 0.0
        else:
            tau_prev  = tau_obs_all[-1]
            tau_ahead = 2 * tau_noisy - tau_prev   # linear extrapolation
            u = np.clip(K_ctrl * (tau_noisy - tau_prev), -U_MAX, U_MAX)
        current_u = u

        obs_frames.append(frame_idx)
        tau_obs_all.append(tau_noisy)
        tau_pred_all.append(tau_ahead)
        ctrl_all.append(u)
        wall_hit_all.append((xw, yw))
        obs_count += 1

    theta += current_u * dt_phys
    x_r   += v * np.cos(theta) * dt_phys
    y_r   += v * np.sin(theta) * dt_phys

    path_x.append(x_r); path_y.append(y_r); path_theta.append(theta)
    frame_idx += 1

N_phys       = len(path_x)
TOTAL_FRAMES = N_phys + 60

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 11))
fig.patch.set_facecolor("#0d0d1a")

ax = fig.add_axes([0.03, 0.07, 0.44, 0.88])
ax.set_facecolor("#0d0d1a")

x_lo = min(path_x) - 7.5
x_hi = max(path_x) + 5.0
y_lo = min(min(path_y) - 5.0, -5.5)
y_hi = obs_y + obs_r + 6.0

ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color="white", fontsize=15)
ax.set_ylabel("y (m)", color="white", fontsize=15)
ax.tick_params(colors="white", labelsize=13)
for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax.grid(True, alpha=0.12, color="white")

# ── Static scene ──────────────────────────────────────────────────────────────
ax.fill_between([x_lo, x_hi], [-0.3, -0.3], [0.3, 0.3],
                color="#2a2a2a", zorder=1)
ax.axhline(0, color="#555", lw=0.8, ls="--", zorder=2)

obs_patch = mpatches.Circle((obs_x, obs_y), obs_r, color="#1a3355", zorder=3)
ax.add_patch(obs_patch)
theta_ring = np.linspace(0, 2*np.pi, 400)
ax.plot(obs_x + obs_r*np.cos(theta_ring),
        obs_y + obs_r*np.sin(theta_ring),
        color="#4da6ff", lw=3.0, zorder=4,
        path_effects=[pe.Stroke(linewidth=5, foreground="#1a3a6a"), pe.Normal()])
ax.scatter(obs_x, obs_y, s=60, color="#4da6ff", zorder=5, marker="+")
ax.annotate(f"obstacle\nr={obs_r}m", (obs_x, obs_y), color="#4da6ff",
            textcoords="offset points", xytext=(8, -18), fontsize=11, ha="left")
ax.scatter(obs_x, obs_y - obs_r, s=100, color="#4da6ff", zorder=5, marker="v")
ax.annotate("closest\npoint", (obs_x, obs_y - obs_r), color="#4da6ff",
            textcoords="offset points", xytext=(8, -28), fontsize=10)

# No-KF label
ax.text(0.5, 0.01, "⚠  NO KALMAN FILTER  — raw τ only",
        transform=ax.transAxes, ha="center", va="bottom",
        color="#ff4444", fontsize=12, fontweight="bold",
        bbox=dict(fc="#1a0000", ec="#ff4444", pad=4, alpha=0.8))

# ── Artists ───────────────────────────────────────────────────────────────────
robot_body,  = ax.plot([], [], "s", ms=18, color="#ff4444",
                        markeredgecolor="#880000", markeredgewidth=2, zorder=8)
robot_trail, = ax.plot([], [], "-",  color="#ff4444", lw=1.5, alpha=0.35, zorder=5)
cam_ray,     = ax.plot([], [], "--", color="#ff9900", lw=2,   alpha=0.85, zorder=6)
feat_dot,    = ax.plot([], [], "o",  ms=13, color="#ff9900",
                        markeredgecolor="white", markeredgewidth=1.2, zorder=10)
wall_trail,  = ax.plot([], [], "o",  ms=5,  color="#ff6600", alpha=0.45, zorder=7)
seg_line,    = ax.plot([], [], "-",  color="white", lw=2.5, alpha=0.9,  zorder=9)
seg_ok,      = ax.plot([], [], "o",  ms=11, color="white", alpha=0.9,  zorder=10)
seg_ok1,     = ax.plot([], [], "*",  ms=18, color="#ff44ff", alpha=0.95, zorder=10)
past_segs = [ax.plot([], [], "-", color="white", lw=0.8,
                     alpha=0.12, zorder=4)[0] for _ in range(N_OBS)]

# ── HUD (figure coords, right panel) ─────────────────────────────────────────
title_t = fig.text(0.75, 0.96, "", ha="center", va="top",
                   color="white", fontsize=15, fontweight="bold")
obs_t   = fig.text(0.50, 0.96, "", ha="left", va="top",
                   color="#ff9900", fontsize=12, fontfamily="monospace")
ctrl_t  = fig.text(0.50, 0.76, "", ha="left", va="top",
                   color="#cc88ff", fontsize=12, fontfamily="monospace")
theta_t = fig.text(0.50, 0.58, "", ha="left", va="top",
                   color="#00ff99", fontsize=12, fontfamily="monospace")

# ── Inset τ plot ──────────────────────────────────────────────────────────────
ax_i = fig.add_axes([0.55, 0.10, 0.43, 0.45])
ax_i.set_facecolor("#11111f")
ax_i.tick_params(colors="white", labelsize=11)
for sp in ax_i.spines.values(): sp.set_edgecolor("#555")
ax_i.set_xlim(-0.5, N_OBS + 0.5)
ax_i.set_ylim(min(tau_obs_all) - 0.5, max(tau_obs_all) + 0.8)
ax_i.set_xlabel("observation k", color="white", fontsize=12)
ax_i.set_ylabel("τ (s)", color="white", fontsize=12)
ax_i.set_title("TTT  —  raw observed τ  (no KF smoothing)",
               color="white", fontsize=12, fontweight="bold")
ax_i.grid(True, alpha=0.18, color="white")
mini_obs,  = ax_i.plot([], [], "o",  ms=7, color="#4da6ff", label="τ_k raw observed")
mini_pred, = ax_i.plot([], [], "^",  ms=9, color="#ff44ff",
                        label="τ_{k+1} extrapolated")
mini_seg2, = ax_i.plot([], [], "--", lw=1.5, color="#ff44ff", alpha=0.55)
now_line   = ax_i.axvline(-1, color="#aaa", lw=1, ls=":", alpha=0.6)
ax_i.legend(fontsize=10, facecolor="#11111f", labelcolor="white",
            loc="upper right", framealpha=0.8)

# ── Main legend ───────────────────────────────────────────────────────────────
legend_els = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#ff4444",
           markersize=12, label="Robot (no KF)"),
    Line2D([0],[0], color="#ff9900", lw=2, ls="--", label="Camera ray τ_k"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#ff9900",
           markersize=9,  label="Circle hit O_k"),
    Line2D([0],[0], color="white", lw=2.5,  label="Segment O_k → O_{k+1}"),
    Line2D([0],[0], marker="*", color="w",  markerfacecolor="#ff44ff",
           markersize=14, label="Extrapolated O_{k+1}"),
    Line2D([0],[0], color="#4da6ff", lw=3,  label="Circular obstacle"),
]
ax.legend(handles=legend_els, loc="lower left", facecolor="#11111f",
          labelcolor="white", fontsize=11, framealpha=0.9)

# ── Animation ─────────────────────────────────────────────────────────────────
def get_k(frame):
    k = -1
    for i, f in enumerate(obs_frames):
        if frame >= f: k = i
    return k

def animate(frame):
    f  = min(frame, N_phys - 1)
    xr, yr, th = path_x[f], path_y[f], path_theta[f]
    th_c = th + np.pi / 2

    robot_body.set_data([xr], [yr])
    robot_trail.set_data(path_x[:f+1], path_y[:f+1])

    k = get_k(frame)
    if k < 0:
        for art in [cam_ray, feat_dot, wall_trail, seg_line, seg_ok, seg_ok1]:
            art.set_data([], [])
        title_t.set_text("Initialising...")
        return []

    xw, yw = wall_hit_all[k]
    cam_ray.set_data([xr, xw], [yr, yw])
    feat_dot.set_data([xw], [yw])
    wall_trail.set_data([wall_hit_all[i][0] for i in range(k+1)],
                        [wall_hit_all[i][1] for i in range(k+1)])

    tau_ah = tau_pred_all[k]
    xw_p = xr + tau_ah * v * np.cos(th_c)
    yw_p = yr + tau_ah * v * np.sin(th_c)
    seg_line.set_data([xw, xw_p], [yw, yw_p])
    seg_ok.set_data([xw], [yw])
    seg_ok1.set_data([xw_p], [yw_p])

    for i, seg in enumerate(past_segs):
        if 0 < i <= k:
            fi     = obs_frames[i-1]
            x0, y0 = wall_hit_all[i-1]
            tp     = tau_pred_all[i-1]
            tc     = path_theta[fi] + np.pi/2
            xp     = path_x[fi] + tp * v * np.cos(tc)
            yp     = path_y[fi] + tp * v * np.sin(tc)
            seg.set_data([x0, xp], [y0, yp])
        else:
            seg.set_data([], [])

    u_k = ctrl_all[k]
    title_t.set_text(f"Observation k = {k}   |   frame {f} / {N_phys}"
                     + ("   ◀ DONE ▶" if frame >= N_phys else ""))
    obs_t.set_text(
        f"τ_k  raw observed  : {tau_obs_all[k]:.3f} s\n"
        f"τ_{{k+1}} extrap    : {tau_pred_all[k]:.3f} s\n"
        f"(no KF smoothing)"
    )
    ctrl_t.set_text(
        f"u_k = K·(τ_k − τ_{{k-1}})\n"
        f"    = {K_ctrl}·(...)\n"
        f"    = {u_k:+.4f} rad"
    )
    theta_t.set_text(f"θ_robot : {np.degrees(th):+.2f}°")

    mini_obs.set_data(range(k+1), tau_obs_all[:k+1])
    mini_pred.set_data([k], [tau_pred_all[k]])
    mini_seg2.set_data([k, k+1], [tau_obs_all[k], tau_pred_all[k]])
    now_line.set_xdata([k, k])

    return (robot_body, robot_trail, cam_ray, feat_dot, wall_trail,
            seg_line, seg_ok, seg_ok1, title_t, obs_t, ctrl_t, theta_t,
            mini_obs, mini_pred, mini_seg2, *past_segs)

ani = animation.FuncAnimation(
    fig, animate,
    frames=TOTAL_FRAMES,
    interval=30,
    blit=False,
    repeat=True
)

plt.show()
