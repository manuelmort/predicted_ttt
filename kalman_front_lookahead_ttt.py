"""
Kalman Filter + Steering Control — Winding Corridor
Dual Forward-Facing Camera, Two Independent KFs
================================================
Light mode — scene only with in-axes HUD (matches kalman_anim_only style)
Control law:  u = K * (τ̂^left_{k+1|k} − τ̂^right_{k+1|k})
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

np.random.seed(42)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_MAIN       = "#ffffff"
BG_HUD        = "#f5f5f5"
CLR_TEXT      = "#111111"
CLR_GRID      = "#cccccc"

CLR_WALL      = "#1a5fa0"
CLR_WALL_GLOW = "#5599cc"
CLR_FILL      = "#ddeeff"
CLR_CENTRE    = "#7aaadd"

CLR_ROBOT     = "#55c47a"
CLR_ROBOT_EDGE= "#1a6b3a"

CLR_LEFT      = "#c85000"       # left ray / hits
CLR_LEFT_TR   = "#e07030"
CLR_RIGHT     = "#007b8a"       # right ray / hits
CLR_RIGHT_TR  = "#33aabb"

CLR_SEG       = "#333333"
CLR_SEG_PAST  = "#aaaaaa"

# ── Tuneable ──────────────────────────────────────────────────────────────────
v              = 2.0
dt_phys        = 0.05
TAU_INTERVAL   = 5
N_OBS          = 60
K_ctrl         = 1.2
U_MAX          = 0.5
noise_std      = 0.05
cam_half_angle = 0.25
corridor_width = 3.0

# ── Build windier corridor ────────────────────────────────────────────────────
n_pts = 600
xs_c  = np.linspace(0, 80, n_pts)
ys_c  = (4.5 * np.sin(xs_c / 10.0)
       + 2.5 * np.sin(xs_c / 5.5  + 1.1)
       + 1.2 * np.sin(xs_c / 3.0  + 2.3))   # third harmonic → tighter wiggles

raw_dx = np.gradient(xs_c)
raw_dy = np.gradient(ys_c)
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
    min_t = np.inf
    hx = hy = None
    for (x1, y1, x2, y2) in wall_segs:
        dx, dy  = x2 - x1, y2 - y1
        denom   = cth * dy - sth * dx
        if abs(denom) < 1e-10:
            continue
        t  = ((x1 - rx_r) * dy  - (y1 - ry_r) * dx) / denom
        s_ = ((x1 - rx_r) * sth - (y1 - ry_r) * cth) / denom
        if t > 0.05 and 0.0 <= s_ <= 1.0 and t < min_t:
            min_t = t
            hx    = rx_r + t * cth
            hy    = ry_r + t * sth
    if hx is None:
        return None
    return hx, hy, min_t / v

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

# ── Robot start ───────────────────────────────────────────────────────────────
start_idx = np.argmin(np.abs(xs_c - 2.0))
x_r   = xs_c[start_idx]
y_r   = ys_c[start_idx]
theta = np.arctan2(ty[start_idx], tx[start_idx])

path_x, path_y, path_theta = [x_r], [y_r], [theta]
obs_frames = []

tau_left_obs_all   = []; tau_left_est_all  = []
tau_left_pred_all  = []; sigma_left_all    = []
wall_left_all      = []

tau_right_obs_all  = []; tau_right_est_all  = []
tau_right_pred_all = []; sigma_right_all    = []
wall_right_all     = []

ctrl_all = []

z_kf_L = None; P_kf_L = None
z_kf_R = None; P_kf_R = None
current_u = 0.0
frame_idx = 0; obs_count = 0

# ── Pre-simulate ──────────────────────────────────────────────────────────────
while obs_count < N_OBS:
    if frame_idx % TAU_INTERVAL == 0:
        th_left  = theta + cam_half_angle
        th_right = theta - cam_half_angle

        hit_left  = ray_wall_hit(x_r, y_r, th_left)
        hit_right = ray_wall_hit(x_r, y_r, th_right)
        if hit_left is None or hit_right is None:
            print(f"Ray miss at obs {obs_count}"); break

        xwl, ywl, tau_left_true  = hit_left
        xwr, ywr, tau_right_true = hit_right

        tau_left_noisy  = tau_left_true  + np.random.normal(0, noise_std)
        tau_right_noisy = tau_right_true + np.random.normal(0, noise_std)

        # KF LEFT
        if z_kf_L is None:
            z_kf_L = np.array([tau_left_noisy, 0.0])
            P_kf_L = np.diag([noise_std**2, 0.5])
        else:
            z_kf_L, P_kf_L = kf_predict(z_kf_L, P_kf_L)
            z_kf_L, P_kf_L = kf_update(z_kf_L, P_kf_L, tau_left_noisy)
        z_ahead_L, _   = kf_predict(z_kf_L, P_kf_L)
        tau_left_ahead = z_ahead_L[0]

        # KF RIGHT
        if z_kf_R is None:
            z_kf_R = np.array([tau_right_noisy, 0.0])
            P_kf_R = np.diag([noise_std**2, 0.5])
        else:
            z_kf_R, P_kf_R = kf_predict(z_kf_R, P_kf_R)
            z_kf_R, P_kf_R = kf_update(z_kf_R, P_kf_R, tau_right_noisy)
        z_ahead_R, _    = kf_predict(z_kf_R, P_kf_R)
        tau_right_ahead = z_ahead_R[0]

        u = 0.0 if obs_count == 0 else \
            np.clip(K_ctrl * (tau_left_ahead - tau_right_ahead), -U_MAX, U_MAX)
        current_u = u

        obs_frames.append(frame_idx)

        tau_left_obs_all.append(tau_left_noisy)
        tau_left_est_all.append(z_kf_L[0])
        tau_left_pred_all.append(tau_left_ahead)
        sigma_left_all.append(np.sqrt(P_kf_L[0, 0]))
        wall_left_all.append((xwl, ywl))

        tau_right_obs_all.append(tau_right_noisy)
        tau_right_est_all.append(z_kf_R[0])
        tau_right_pred_all.append(tau_right_ahead)
        sigma_right_all.append(np.sqrt(P_kf_R[0, 0]))
        wall_right_all.append((xwr, ywr))

        ctrl_all.append((u, tau_left_ahead, tau_right_ahead))
        obs_count += 1

    theta += current_u * dt_phys
    x_r   += v * np.cos(theta) * dt_phys
    y_r   += v * np.sin(theta) * dt_phys
    path_x.append(x_r); path_y.append(y_r); path_theta.append(theta)
    frame_idx += 1

N_phys       = len(path_x)
TOTAL_FRAMES = N_phys + 60

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_MAIN)

ax = fig.add_axes([0.06, 0.07, 0.91, 0.85])
ax.set_facecolor(BG_MAIN)

pad  = 2.5
x_lo = min(path_x) - pad;   x_hi = max(path_x) + pad
y_lo = min(min(path_y), np.min(ly), np.min(ry)) - pad
y_hi = max(max(path_y), np.max(ly), np.max(ry)) + pad

ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=CLR_TEXT, fontsize=13)
ax.set_ylabel("y (m)", color=CLR_TEXT, fontsize=13)
ax.tick_params(colors=CLR_TEXT, labelsize=11)
for sp in ax.spines.values(): sp.set_edgecolor("#aaaaaa")
ax.grid(True, alpha=0.35, color=CLR_GRID)

# ── Static corridor ───────────────────────────────────────────────────────────
fill_x = np.concatenate([lx, rx[::-1]])
fill_y = np.concatenate([ly, ry[::-1]])
ax.fill(fill_x, fill_y, color=CLR_FILL, alpha=0.55, zorder=2)
ax.plot(xs_c, ys_c, color=CLR_CENTRE, lw=1.0, ls="--", alpha=0.45, zorder=3)
for wall_arr_x, wall_arr_y in [(lx, ly), (rx, ry)]:
    ax.plot(wall_arr_x, wall_arr_y, color=CLR_WALL, lw=2.5, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground=CLR_WALL_GLOW),
                          pe.Normal()])

# Start marker
ax.scatter(path_x[0], path_y[0], s=110, color=CLR_ROBOT,
           marker="D", zorder=12, edgecolors=CLR_ROBOT_EDGE, linewidths=1.5)
ax.annotate("START", (path_x[0], path_y[0]), color=CLR_TEXT,
            textcoords="offset points", xytext=(6, 8), fontsize=10,
            fontweight="bold")

# ── Animated artists ──────────────────────────────────────────────────────────
robot_body,  = ax.plot([], [], "s", ms=14, color=CLR_ROBOT,
                        markeredgecolor=CLR_ROBOT_EDGE, markeredgewidth=2, zorder=10)
robot_trail, = ax.plot([], [], "-", color=CLR_ROBOT, lw=1.8, alpha=0.40, zorder=5)
heading_q    = ax.quiver([0], [0], [0], [0],
                          color=CLR_ROBOT, scale=18, scale_units="xy",
                          angles="xy", width=0.012, headwidth=4,
                          headlength=5, zorder=11)

# Left ray — burnt orange
cam_ray_L, = ax.plot([], [], "--", color=CLR_LEFT,    lw=2,  alpha=0.85, zorder=7)
feat_L,    = ax.plot([], [], "o",  ms=12, color=CLR_LEFT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=11)
trail_L,   = ax.plot([], [], "o",  ms=4,  color=CLR_LEFT_TR, alpha=0.40, zorder=6)

# Right ray — teal
cam_ray_R, = ax.plot([], [], "--", color=CLR_RIGHT,    lw=2,  alpha=0.85, zorder=7)
feat_R,    = ax.plot([], [], "o",  ms=12, color=CLR_RIGHT,
                      markeredgecolor="#333333", markeredgewidth=1.2, zorder=11)
trail_R,   = ax.plot([], [], "o",  ms=4,  color=CLR_RIGHT_TR, alpha=0.40, zorder=6)

seg_line, = ax.plot([], [], "-",  color=CLR_SEG,      lw=2.5, alpha=0.9,  zorder=9)
past_segs = [ax.plot([], [], "-", color=CLR_SEG_PAST, lw=0.8,
                     alpha=0.20, zorder=4)[0] for _ in range(N_OBS)]

# ── HUD — inside axes, top-left ───────────────────────────────────────────────
hud_kw = dict(transform=ax.transAxes, va="top", fontfamily="monospace",
              fontsize=10, bbox=dict(boxstyle="round,pad=0.4",
                                     facecolor=BG_HUD, edgecolor="#bbbbbb",
                                     alpha=0.88))
title_t = ax.text(0.01, 0.99, "", color=CLR_TEXT,   **hud_kw)
obs_t   = ax.text(0.01, 0.91, "", color=CLR_LEFT,   **hud_kw)
obs_r_t = ax.text(0.01, 0.74, "", color=CLR_RIGHT,  **hud_kw)
ctrl_t  = ax.text(0.01, 0.57, "", color="#444444",  **hud_kw)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_els = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=CLR_ROBOT,
           markersize=11, label="Robot", markeredgecolor=CLR_ROBOT_EDGE),
    Line2D([0],[0], color=CLR_WALL,  lw=2.5, label="Corridor walls"),
    Line2D([0],[0], color=CLR_LEFT,  lw=2, ls="--",
           label=r"Left ray → $O^{left}_k$"),
    Line2D([0],[0], color=CLR_RIGHT, lw=2, ls="--",
           label=r"Right ray → $O^{right}_k$"),
    Line2D([0],[0], color=CLR_SEG,   lw=2.5,
           label=r"Segment $O^{left}_k$ → $O^{right}_k$"),
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

    u_k, tau_la, tau_ra = ctrl_all[k]

    title_t.set_text(
        f"k = {k:3d}   frame {f}/{N_phys}"
        + ("   ◀ DONE ▶" if frame >= N_phys else ""))

    obs_t.set_text(
        f"LEFT ray\n"
        f"  τ obs  : {tau_left_obs_all[k]:.4f} s\n"
        f"  τ̂ post : {tau_left_est_all[k]:.4f} s\n"
        f"  τ̂ pred : {tau_left_pred_all[k]:.4f} s   σ={sigma_left_all[k]:.4f}")

    obs_r_t.set_text(
        f"RIGHT ray\n"
        f"  τ obs  : {tau_right_obs_all[k]:.4f} s\n"
        f"  τ̂ post : {tau_right_est_all[k]:.4f} s\n"
        f"  τ̂ pred : {tau_right_pred_all[k]:.4f} s   σ={sigma_right_all[k]:.4f}")

    ctrl_t.set_text(
        f"u = K·(τ̂^L_{{k+1|k}} − τ̂^R_{{k+1|k}})\n"
        f"  Δτ = {tau_la - tau_ra:+.4f}\n"
        f"  u_k = {u_k:+.4f} rad\n"
        f"  θ   = {np.degrees(th):+.2f}°")

    return (robot_body, robot_trail, heading_q,
            cam_ray_L, feat_L, trail_L,
            cam_ray_R, feat_R, trail_R,
            seg_line, title_t, obs_t, obs_r_t, ctrl_t,
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
