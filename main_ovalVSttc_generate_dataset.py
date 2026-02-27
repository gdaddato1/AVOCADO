#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from actors import AVOCADO_Actor

from evaluate_dataset_avocado import (
    _distance_series_to_humans,
    _goal_reached,
    _mean_curvature,
    _path_length,
    _projected_time_to_collision,
    _safe_nanmean,
)


try:
    from dataset_generation_hooks import generate_human_paths, make_robot_start_goal
except Exception:
    try:
        from generate_human_paths import generate_human_paths
        from make_robot_start_goal import make_robot_start_goal
    except Exception:
        def generate_human_paths(*args, **kwargs):
            raise NotImplementedError(
                "Provide generate_human_paths via dataset_generation_hooks.py, "
                "or keep local modules generate_human_paths.py and make_robot_start_goal.py available."
            )

        def make_robot_start_goal(*args, **kwargs):
            raise NotImplementedError(
                "Provide make_robot_start_goal via dataset_generation_hooks.py, "
                "or keep local modules generate_human_paths.py and make_robot_start_goal.py available."
            )


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class OpinionParams:
    dr: float = 2.4
    alpha_r: float = 0.3
    gamma_r: float = 10.0
    Rr: float = 6.0
    kr: float = 1.5
    beta_r: float = np.pi / 4.0
    u_max: float = 2.0
    u_min: float = 0.0
    n: int = 7
    tau_u: float = 1.0


def get_opinion_params(overrides: Dict[str, Any] | None = None) -> OpinionParams:
    par = OpinionParams()
    if not overrides:
        return par
    valid = set(par.__dataclass_fields__.keys())
    for key, value in overrides.items():
        if key in valid and value is not None:
            setattr(par, key, value)
    return par


def attention_dynamics_kappa(par: OpinionParams, chi: float, kappa: float) -> float:
    num = (par.Rr * kappa) ** par.n
    den = num + chi**par.n
    return par.u_min + (par.u_max - par.u_min) * (num / den)


def attention_dynamics_ttc(par: OpinionParams, ttc: float) -> float:
    if np.isinf(ttc):
        return par.u_max
    num = (par.Rr * ttc) ** par.n
    den = num + 1.0
    return par.u_min + (par.u_max - par.u_min) * (num / den)


def _projected_ttc_scalar(rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float) -> float:
    c = float(np.dot(rel_pos, rel_pos) - radius**2)
    if c <= 0:
        return 0.0
    a = float(np.dot(rel_vel, rel_vel))
    b = float(2.0 * np.dot(rel_pos, rel_vel))
    if a < 1e-12:
        return np.nan
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.nan
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return np.nan
    return float(min(candidates))


def compute_metrics(
    robot_traj: np.ndarray,
    human_traj: np.ndarray,
    dt: float,
    scenario_type: str,
    collision_distance: float = 1.0,
) -> Dict[str, Any]:
    T = min(robot_traj.shape[1], human_traj.shape[1])
    robot_traj = robot_traj[:, :T]
    human_traj = human_traj[:, :T]

    metrics: Dict[str, Any] = {}
    dists = np.sqrt(np.sum((robot_traj - human_traj) ** 2, axis=0))
    metrics["min_distance"] = float(np.min(dists))

    if T >= 2:
        headings = np.arctan2(np.diff(robot_traj[1, :]), np.diff(robot_traj[0, :]))
        headings = np.concatenate([headings, [headings[-1]]])
    else:
        headings = np.zeros(T)

    vec_rh = human_traj - robot_traj
    angles_rh = np.arctan2(vec_rh[1, :], vec_rh[0, :])
    angle_diffs = wrap_to_pi(angles_rh - headings)
    in_cone = (angle_diffs >= -np.pi / 6.0) & (angle_diffs <= np.pi / 6.0)
    metrics["min_distance_in_cone"] = float(np.min(dists[in_cone])) if np.any(in_cone) else np.nan

    if T >= 2:
        steps = np.sqrt(np.sum(np.diff(robot_traj, axis=1) ** 2, axis=0))
        actual_path_length = float(np.sum(steps))
    else:
        actual_path_length = 0.0

    robot_start = robot_traj[:, 0]
    robot_goal = robot_traj[:, -1]
    straight_line_distance = float(np.linalg.norm(robot_goal - robot_start))
    metrics["path_efficiency"] = (
        straight_line_distance / actual_path_length if actual_path_length > 1e-12 else 0.0
    )
    metrics["path_length"] = actual_path_length
    metrics["time_to_goal"] = float(T * dt)

    threshold_dist = collision_distance
    interaction_times = dists < threshold_dist
    metrics["interaction_time"] = float(np.sum(interaction_times) * dt)

    crossed = np.where(dists < threshold_dist)[0]
    metrics["time_to_collision_threshold"] = (
        float(crossed[0] * dt) if crossed.size > 0 else np.nan
    )

    ttc_values: List[float] = []
    ttc_proj_abs_values: List[float] = []
    best_abs_ttc = np.inf

    if T >= 2:
        vel_r = np.diff(robot_traj, axis=1) / dt
        vel_h = np.diff(human_traj, axis=1) / dt
        for t in range(T - 1):
            rel_pos = human_traj[:, t] - robot_traj[:, t]
            rel_vel = vel_h[:, t] - vel_r[:, t]
            ttc_now = _projected_ttc_scalar(rel_pos, rel_vel, threshold_dist)
            if not np.isnan(ttc_now):
                ttc_values.append(float(ttc_now))
                abs_ttc = t * dt + ttc_now
                ttc_proj_abs_values.append(float(abs_ttc))
                best_abs_ttc = min(best_abs_ttc, abs_ttc)

    metrics["avg_ttc"] = float(np.mean(ttc_values)) if ttc_values else np.nan
    metrics["min_ttc"] = float(np.min(ttc_values)) if ttc_values else np.nan
    metrics["min_ttc_proj"] = float(best_abs_ttc) if np.isfinite(best_abs_ttc) else np.nan
    metrics["avg_ttc_proj"] = float(np.mean(ttc_proj_abs_values)) if ttc_proj_abs_values else np.nan

    dx = np.gradient(robot_traj[0, :], dt)
    dy = np.gradient(robot_traj[1, :], dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    denom = (dx**2 + dy**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    metrics["avg_curvature"] = float(np.mean(curvature))
    metrics["max_curvature"] = float(np.max(curvature))

    vel_r = np.diff(robot_traj, axis=1) / dt if T >= 2 else np.zeros((2, 0))
    if vel_r.shape[1] >= 1:
        headings_r = np.arctan2(vel_r[1, :], vel_r[0, :])
        heading_changes = np.diff(headings_r)
        metrics["mean_heading_change"] = float(np.mean(np.abs(heading_changes))) if heading_changes.size else 0.0
        metrics["max_heading_change"] = float(np.max(np.abs(heading_changes))) if heading_changes.size else 0.0
    else:
        metrics["mean_heading_change"] = 0.0
        metrics["max_heading_change"] = 0.0

    if T < 4:
        metrics["path_smoothness"] = 1.0
    else:
        vel = np.diff(robot_traj, axis=1) / dt
        speed = np.sqrt(np.sum(vel**2, axis=0))
        theta = np.arctan2(vel[1, :], vel[0, :])
        dtheta = wrap_to_pi(np.diff(theta))
        mean_curv = float(np.mean(np.abs(dtheta))) if dtheta.size else 0.0
        acc = np.diff(speed) / dt
        metrics["path_smoothness"] = float(1.0 / (1.0 + mean_curv + np.std(acc)))

    line_vec = robot_goal - robot_start
    line_len = float(np.linalg.norm(line_vec))
    line_dir = line_vec / (line_len + 1e-6)
    proj_lengths = (robot_traj - robot_start.reshape(2, 1)).T @ line_dir
    proj_points = robot_start.reshape(2, 1) + np.outer(line_dir, proj_lengths)
    deviations = np.sqrt(np.sum((robot_traj - proj_points) ** 2, axis=0))
    metrics["mean_deviation"] = float(np.mean(deviations))
    metrics["max_deviation"] = float(np.max(deviations))

    metrics["safety_efficiency_tradeoff"] = (
        metrics["min_distance"] / (metrics["path_length"] + 1e-12)
    )

    avg_distance = float(np.mean(dists))
    w_comfort = 0.6
    w_efficiency = 0.4
    metrics["weighted_comfort_efficiency"] = float(
        w_comfort * (avg_distance / 1.5) + w_efficiency * metrics["path_efficiency"]
    )
    metrics["collision_distance"] = float(collision_distance)

    metrics["scenario"] = scenario_type
    metrics["variant"] = scenario_type
    metrics["robot_traj"] = robot_traj
    metrics["human_traj"] = human_traj
    return metrics


def simulate_social_nav(
    robot_start: np.ndarray,
    robot_goal: np.ndarray,
    human_waypoints: np.ndarray,
    attention_mode: str,
    v_human: float,
    v_robot: float,
    dt: float,
    max_time: float,
) -> Dict[str, Any]:
    par = get_opinion_params()

    xr = robot_start.astype(float).copy()
    xr5 = robot_start.astype(float).copy()
    xr6 = robot_start.astype(float).copy()
    xrg = robot_goal.astype(float).copy()

    xh = human_waypoints[:, 0].astype(float).copy()
    current_wp = 0

    theta = np.arctan2(xrg[1] - xr[1], xrg[0] - xr[0])
    theta5 = theta
    theta6 = theta

    b1 = 2.0
    eps_oval = b1 / 4.0
    b2 = eps_oval * b1
    nu = -0.5
    alpha1 = 0.5
    alpha2 = 6.5
    xt = b1 * 0.75
    k_att = 0.7
    k_rep = 0.6

    robot_vel6 = np.zeros(2, dtype=float)
    avocado_actor = AVOCADO_Actor(
        agent_radius=0.2,
        timestep=dt,
        alpha=[100.0],
        a=0.3,
        c=0.7,
        d=2.0,
        kappa=14.15,
        epsilon=3.22,
        delta=0.57,
        bias=[0.0],
    )

    i2 = min(1, human_waypoints.shape[1] - 1)
    theta_h = np.arctan2(human_waypoints[1, i2] - xh[1], human_waypoints[0, i2] - xh[0])

    z = -0.01
    u = 0.01

    robot_traj = [xr.copy()]
    robot_traj5 = [xr5.copy()]
    robot_traj6 = [xr6.copy()]
    human_traj = [xh.copy()]
    time_vec = [0.0]

    z_vals = [z]
    u_vals = [u]
    theta_vals = [theta]
    eta_h_vals: List[float] = []

    theta_vals5 = [theta5]
    theta_vals6 = [theta6]
    rho5_vals = [0.0]

    z_hat_body_vals: List[float] = []

    t = 0.0
    while np.linalg.norm(xr - xrg) > 0.5 and t < max_time:
        eta_body = wrap_to_pi(np.arctan2(xr[1] - xh[1], xr[0] - xh[0]) - theta_h)

        chi = float(np.linalg.norm(xh - xr))
        kappa = max(np.cos(eta_body), 0.0)

        desired_angle = wrap_to_pi(np.arctan2(xrg[1] - xr[1], xrg[0] - xr[0]) - theta)
        phi_r = desired_angle

        z_hat_body = np.sin(-eta_body)

        eta_h_vals.append(float(eta_body))

        eps_ttc = 1e-6
        v_r = v_robot
        v_h = v_human

        e_rh = (xh - xr) / max(chi, eps_ttc)
        eta_r = wrap_to_pi(-theta + np.arctan2(e_rh[1], e_rh[0]))
        eta_h_ttc = wrap_to_pi(-theta_h + np.arctan2(-e_rh[1], -e_rh[0]))
        v_proj = v_r * np.cos(eta_r) + v_h * np.cos(eta_h_ttc)
        ttc1 = chi / v_proj if v_proj > eps_ttc else np.inf

        br = 0.0
        z_dot = -par.dr * z + u * np.tanh(par.alpha_r * z + par.gamma_r * z_hat_body + br)

        if attention_mode == "kappa":
            u_dot = (-u + attention_dynamics_kappa(par, chi, kappa)) / par.tau_u
        elif attention_mode == "ttc":
            u_dot = (-u + attention_dynamics_ttc(par, ttc1)) / par.tau_u
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        z += z_dot * dt
        u += u_dot * dt

        omega = par.kr * np.sin(par.beta_r * np.tanh(z) + phi_r)

        theta = wrap_to_pi(theta + omega * dt)

        if np.linalg.norm(xh - human_waypoints[:, current_wp]) < 0.2:
            current_wp = min(current_wp + 1, human_waypoints.shape[1] - 1)

        target_wp = human_waypoints[:, current_wp]
        th_des = np.arctan2(target_wp[1] - xh[1], target_wp[0] - xh[0])
        theta_h = wrap_to_pi(theta_h + 2.0 * wrap_to_pi(th_des - theta_h) * dt)
        xh = xh + v_human * np.array([np.cos(theta_h), np.sin(theta_h)]) * dt

        robot_traj.append(xr.copy())
        robot_traj5.append(xr5.copy())
        robot_traj6.append(xr6.copy())
        human_traj.append(xh.copy())

        time_vec.append(time_vec[-1] + dt)
        z_vals.append(z)
        u_vals.append(u)

        theta_vals.append(theta)

        z_hat_body_vals.append(float(z_hat_body))
        theta_vals5.append(theta5)
        theta_vals6.append(theta6)

        if np.linalg.norm(xr - xrg) > 0.5:
            xr[0] += v_robot * np.cos(theta) * dt
            xr[1] += v_robot * np.sin(theta) * dt

        if np.linalg.norm(xr5 - xrg) > 0.5:
            gamma = -1.0 if z >= 0.0 else 1.0

            dx_goal5 = xrg[0] - xr5[0]
            dy_goal5 = xrg[1] - xr5[1]
            Fx_att = k_att * dx_goal5
            Fy_att = k_att * dy_goal5

            theta_robot5 = np.arctan2(xh[1] - xr5[1], xh[0] - xr5[0])

            xr1_5 = xr5[0]
            xr2_5 = xr5[1]
            xr1rot = (xr1_5 - xh[0]) * np.cos(-theta_robot5) - (xr2_5 - xh[1]) * np.sin(-theta_robot5) + xt
            xr2rot = (xr1_5 - xh[0]) * np.sin(-theta_robot5) + (xr2_5 - xh[1]) * np.cos(-theta_robot5)

            rho = 1.0 - (xr1rot / b1) ** 2 - (xr2rot / b2) ** 2 * np.exp(nu * xr1rot)
            rho5_current = float(rho)

            sqrt_arg = (
                (b1**2 * eps_oval**2 * xr1rot**2)
                - (2.0 * b1**2 * eps_oval**2 * xr1rot * xt)
                + (b1**2 * eps_oval**2 * xt**2)
                + (xr2rot**2 * np.exp(nu * xr1rot) * b1**2)
                - (np.exp(nu * xr1rot) * xt**2 * xr2rot**2)
            )
            r_oval = (
                (
                    eps_oval * xr1rot * xt
                    - (eps_oval * xt**2)
                    + np.sqrt(max(float(sqrt_arg), 0.0))
                )
                / (b1**2 - xt**2)
                * b1
                / eps_oval
            )
            d = xt * (1.0 - r_oval / b1)
            x1_bar = xr1rot - d

            if rho > 0.0:
                xfb = np.array([rho * (xr1rot - xt), rho * xr2rot])
                xff = np.array(
                    [
                        -(b1 / b2) * xr2rot * np.exp((nu / 2.0) * x1_bar),
                        ((b2 / b1) * x1_bar * np.exp((-nu / 2.0) * x1_bar))
                        + (xr2rot**2 * np.exp((nu / 2.0) * x1_bar) * nu * (b1 / (2.0 * b2))),
                    ]
                )
            else:
                xfb = np.array([0.0, 0.0])
                xff = np.array([0.0, 0.0])

            alpha_mat = np.array([[alpha1, 0.0], [0.0, alpha2]])
            F_rep = gamma * xff + alpha_mat @ xfb
            Fx_rep = k_rep * F_rep[0]
            Fy_rep = k_rep * F_rep[1]

            omega5 = wrap_to_pi(np.arctan2(Fy_att + Fy_rep, Fx_att + Fx_rep) - theta5)
            theta5 = wrap_to_pi(theta5 + omega5 * dt)

            xr5[0] += v_robot * np.cos(theta5) * dt
            xr5[1] += v_robot * np.sin(theta5) * dt
        else:
            rho5_current = rho5_vals[-1]

        if np.linalg.norm(xr6 - xrg) > 0.5:
            vh6 = v_human * np.array([np.cos(theta_h), np.sin(theta_h)], dtype=float)
            cmd6 = avocado_actor.act(
                agent_positions=xr6.reshape(1, 2),
                other_positions=xh.reshape(1, 2),
                agent_velocities=robot_vel6.reshape(1, 2),
                other_velocities=vh6.reshape(1, 2),
                agent_goals=xrg.reshape(1, 2),
                max_vel=v_robot,
            )[0]
            robot_vel6 = np.asarray(cmd6, dtype=float).reshape(2)
            xr6 = xr6 + robot_vel6 * dt
            if np.linalg.norm(robot_vel6) > 1e-8:
                theta6 = float(np.arctan2(robot_vel6[1], robot_vel6[0]))
        else:
            robot_vel6 = np.zeros(2, dtype=float)

        rho5_vals.append(float(rho5_current))

        if current_wp == human_waypoints.shape[1] - 1 and np.linalg.norm(xh - human_waypoints[:, -1]) < 0.2:
            theta_h = 0.0

        t += dt

    result = {
        "robot_traj": np.column_stack(robot_traj),
        "robot_traj5": np.column_stack(robot_traj5),
        "robot_traj_avocado": np.column_stack(robot_traj6),
        "human_traj": np.column_stack(human_traj),
        "time": np.asarray(time_vec),
        "z": np.asarray(z_vals),
        "u": np.asarray(u_vals),
        "theta": np.asarray(theta_vals),
        "theta5": np.asarray(theta_vals5),
        "theta_avocado": np.asarray(theta_vals6),
        "eta_h": np.asarray(eta_h_vals),
        "z_hat_body": np.asarray(z_hat_body_vals),
        "rho5": np.asarray(rho5_vals),
    }
    return result


def _mean_metric(metrics: List[Dict[str, Any]], key: str) -> float:
    vals = [m[key] for m in metrics if key in m and not np.isnan(m[key])]
    return float(np.mean(vals)) if vals else np.nan


def _print_improvement(
    label: str,
    baseline: Dict[str, float],
    candidate: Dict[str, float],
) -> None:
    b_path = baseline["path"]
    b_dist = baseline["dist"]
    b_smooth = baseline["smooth"]
    b_ttc = baseline["ttc"]
    b_curv = baseline["curv"]

    c_path = candidate["path"]
    c_dist = candidate["dist"]
    c_smooth = candidate["smooth"]
    c_ttc = candidate["ttc"]
    c_curv = candidate["curv"]

    imp_path = 100.0 * (b_path - c_path) / max(abs(b_path), np.finfo(float).eps)
    imp_dist = 100.0 * (c_dist - b_dist) / max(abs(b_dist), np.finfo(float).eps)
    imp_smooth = 100.0 * (c_smooth - b_smooth) / max(abs(b_smooth), np.finfo(float).eps)
    imp_ttc = 100.0 * (c_ttc - b_ttc) / max(abs(b_ttc), np.finfo(float).eps)
    imp_curv = 100.0 * (b_curv - c_curv) / max(abs(b_curv), np.finfo(float).eps)

    print(label)
    print(
        "  Path Length: {:+.2f}% | Min Distance: {:+.2f}% | Smoothness: {:+.2f}% | "
        "Min TTC Proj: {:+.2f}% | Avg Curvature: {:+.2f}%".format(
            imp_path, imp_dist, imp_smooth, imp_ttc, imp_curv
        )
    )
    print(
        "  Values (candidate): "
        f"Path {c_path:.4f} | "
        f"MinDist {c_dist:.4f} | "
        f"Smooth {c_smooth:.4f} | "
        f"MinTTCproj {c_ttc:.4f} | "
        f"Curv {c_curv:.4f}"
    )


def _plot_legacy_comparison_samples(
    *,
    sample_keys: List[Tuple[int, float, float]],
    sample_store: Dict[Tuple[int, float, float], Dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for key in sample_keys:
        rec = sample_store[key]
        scenario = rec["scenario"]
        v_h = rec["v_human"]
        v_r = rec["v_robot"]
        robot_start = rec["robot_start"]
        robot_goal = rec["robot_goal"]

        fig, ax = plt.subplots(figsize=(7, 7))

        h = rec["human_traj"]
        ax.plot(h[0, :], h[1, :], color="tab:blue", linewidth=1.8, label="human")

        ax.plot(rec["traj1_kappa"][0, :], rec["traj1_kappa"][1, :], color="tab:orange", linewidth=2.0, label="traj1 + kappa")
        ax.plot(rec["traj1_ttc"][0, :], rec["traj1_ttc"][1, :], color="tab:green", linewidth=2.0, label="traj1 + ttc")
        ax.plot(rec["traj5_kappa"][0, :], rec["traj5_kappa"][1, :], color="tab:red", linewidth=2.0, label="traj5 (ovals) + kappa")
        ax.plot(rec["avocado"][0, :], rec["avocado"][1, :], color="tab:purple", linewidth=2.0, label="avocado")

        ax.scatter(robot_start[0], robot_start[1], color="black", marker="^", s=60, label="robot start")
        ax.scatter(robot_goal[0], robot_goal[1], color="black", marker="*", s=90, label="robot goal")

        ax.set_title(f"sample {key[0]} | {scenario} | vh={v_h:.2f} vr={v_r:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        out_name = f"comparison_sample-{key[0]:03d}_{scenario}_vh-{v_h:.2f}_vr-{v_r:.2f}.png"
        fig.tight_layout()
        fig.savefig(output_dir / out_name, dpi=150)
        plt.close(fig)


def _animate_traj5_with_oval(
    *,
    human_traj: np.ndarray,
    traj5: np.ndarray,
    robot_start: np.ndarray,
    robot_goal: np.ndarray,
    output_path: Path,
    fps: int = 20,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    b1 = 2.0
    eps_oval = b1 / 4.0
    b2 = eps_oval * b1
    nu = -0.5
    xt = b1 * 0.75

    T = min(human_traj.shape[1], traj5.shape[1])
    human = human_traj[:, :T]
    robot = traj5[:, :T]
    theta_h_series = np.zeros(T, dtype=float)
    if T >= 2:
        d_h = np.diff(human, axis=1)
        theta_h_series[1:] = np.arctan2(d_h[1, :], d_h[0, :])
        theta_h_series[0] = theta_h_series[1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(human[0, :], human[1, :], color="tab:blue", alpha=0.25, linewidth=1.0)
    ax.plot(robot[0, :], robot[1, :], color="tab:red", alpha=0.25, linewidth=1.0)

    human_line, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="human")
    robot_line, = ax.plot([], [], color="tab:red", linewidth=2.0, label="traj5 (ovals)")
    human_dot, = ax.plot([], [], "o", color="tab:blue", ms=6)
    robot_dot, = ax.plot([], [], "o", color="tab:red", ms=6)
    oval_line, = ax.plot([], [], color="tab:purple", linewidth=1.5, label="corresponding oval")

    grid_x1 = np.linspace(-b1, b1, 15)
    grid_x2 = np.linspace(-b2, b2, 11)
    gx1, gx2 = np.meshgrid(grid_x1, grid_x2)
    gx1 = gx1.ravel()
    gx2 = gx2.ravel()
    rho_grid = 1.0 - (gx1 / b1) ** 2 - (gx2 / b2) ** 2 * np.exp(nu * gx1)
    inside = rho_grid > 0.0
    x1_in = gx1[inside]
    x2_in = gx2[inside]
    qx0 = np.zeros_like(x1_in)
    qy0 = np.zeros_like(x1_in)
    qu0 = np.zeros_like(x1_in)
    qv0 = np.zeros_like(x1_in)
    vortex_quiver = ax.quiver(
        qx0,
        qy0,
        qu0,
        qv0,
        color="tab:gray",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=3.5,
        width=0.0025,
    )

    ax.scatter(robot_start[0], robot_start[1], color="black", marker="^", s=60, label="robot start")
    ax.scatter(robot_goal[0], robot_goal[1], color="black", marker="*", s=90, label="robot goal")

    all_x = np.concatenate((human[0, :], robot[0, :], [robot_start[0], robot_goal[0]]))
    all_y = np.concatenate((human[1, :], robot[1, :], [robot_start[1], robot_goal[1]]))
    pad = 1.0
    ax.set_xlim(np.min(all_x) - pad, np.max(all_x) + pad)
    ax.set_ylim(np.min(all_y) - pad, np.max(all_y) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

    x1 = np.linspace(-b1, b1, 240)

    def _oval_world_points(hx: float, hy: float, theta_orient: float):
        inside = np.maximum(0.0, 1.0 - (x1 / b1) ** 2)
        x2_abs = b2 * np.sqrt(inside * np.exp(-nu * x1))

        x1_full = np.concatenate((x1, x1[::-1]))
        x2_full = np.concatenate((x2_abs, -x2_abs[::-1]))

        u = x1_full - xt
        v = x2_full
        dx = u * np.cos(theta_orient) - v * np.sin(theta_orient)
        dy = u * np.sin(theta_orient) + v * np.cos(theta_orient)
        return hx + dx, hy + dy

    def _vortex_field_world(hx: float, hy: float, theta_orient: float):
        sqrt_arg = (
            (b1**2 * eps_oval**2 * x1_in**2)
            - (2.0 * b1**2 * eps_oval**2 * x1_in * xt)
            + (b1**2 * eps_oval**2 * xt**2)
            + (x2_in**2 * np.exp(nu * x1_in) * b1**2)
            - (np.exp(nu * x1_in) * xt**2 * x2_in**2)
        )
        r_oval = (
            (
                eps_oval * x1_in * xt
                - (eps_oval * xt**2)
                + np.sqrt(np.maximum(sqrt_arg, 0.0))
            )
            / (b1**2 - xt**2)
            * b1
            / eps_oval
        )
        d = xt * (1.0 - r_oval / b1)
        x1_bar = x1_in - d

        vortex_x = -(b1 / b2) * x2_in * np.exp((nu / 2.0) * x1_bar)
        vortex_y = (
            (b2 / b1) * x1_bar * np.exp((-nu / 2.0) * x1_bar)
            + (x2_in**2 * np.exp((nu / 2.0) * x1_bar) * nu * (b1 / (2.0 * b2)))
        )

        pos_u = x1_in - xt
        pos_v = x2_in
        pos_x = hx + pos_u * np.cos(theta_orient) - pos_v * np.sin(theta_orient)
        pos_y = hy + pos_u * np.sin(theta_orient) + pos_v * np.cos(theta_orient)

        vec_x = vortex_x * np.cos(theta_orient) - vortex_y * np.sin(theta_orient)
        vec_y = vortex_x * np.sin(theta_orient) + vortex_y * np.cos(theta_orient)
        return pos_x, pos_y, vec_x, vec_y

    def _update(frame: int):
        human_line.set_data(human[0, : frame + 1], human[1, : frame + 1])
        robot_line.set_data(robot[0, : frame + 1], robot[1, : frame + 1])
        human_dot.set_data([human[0, frame]], [human[1, frame]])
        robot_dot.set_data([robot[0, frame]], [robot[1, frame]])

        theta_orient = float(theta_h_series[frame] + np.pi)

        ox, oy = _oval_world_points(
            hx=float(human[0, frame]),
            hy=float(human[1, frame]),
            theta_orient=theta_orient,
        )
        oval_line.set_data(ox, oy)

        qx, qy, qu, qv = _vortex_field_world(
            hx=float(human[0, frame]),
            hy=float(human[1, frame]),
            theta_orient=theta_orient,
        )
        vortex_quiver.set_offsets(np.column_stack((qx, qy)))
        vortex_quiver.set_UVC(qu, qv)
        ax.set_title(f"traj5 + oval | t={frame}")
        return human_line, robot_line, human_dot, robot_dot, oval_line, vortex_quiver

    ani = animation.FuncAnimation(fig, _update, frames=T, interval=1000 / max(1, fps), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)


def _human_state_for_opinion(
    human_positions: np.ndarray,
    human_velocities: np.ndarray,
    step: int,
    robot_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if human_positions.ndim == 2:
        idx = min(step, human_positions.shape[0] - 1)
        pos = np.asarray(human_positions[idx], dtype=float)
        if step < human_velocities.shape[0]:
            vel = np.asarray(human_velocities[idx], dtype=float)
        else:
            vel = np.zeros(2, dtype=float)
        return pos.reshape(2), vel.reshape(2)

    if human_positions.ndim == 3:
        idx = min(step, human_positions.shape[0] - 1)
        pos_all = np.asarray(human_positions[idx], dtype=float)
        if step < human_velocities.shape[0]:
            vel_all = np.asarray(human_velocities[idx], dtype=float)
        else:
            vel_all = np.zeros_like(pos_all)
        dists = np.linalg.norm(pos_all - robot_pos[None, :], axis=1)
        pick = int(np.argmin(dists))
        return pos_all[pick].reshape(2), vel_all[pick].reshape(2)

    raise ValueError("Unexpected human_positions shape. Expected (T,2) or (T,H,2).")


def _run_opinion_rollout_from_dataset(
    *,
    human_positions: np.ndarray,
    human_velocities: np.ndarray,
    robot_start: np.ndarray,
    robot_goal: np.ndarray,
    dt: float,
    robot_speed: float,
    opinion_params: OpinionParams,
    attention_mode: str,
    collision_distance: float,
    goal_tolerance: float,
    extra_steps: int,
) -> np.ndarray:
    total_steps = int(human_positions.shape[0]) + max(0, int(extra_steps))

    xr = np.asarray(robot_start, dtype=float).reshape(2).copy()
    xrg = np.asarray(robot_goal, dtype=float).reshape(2).copy()

    direction = xrg - xr
    theta = float(np.arctan2(direction[1], direction[0]))

    xh0, vh0 = _human_state_for_opinion(human_positions, human_velocities, 0, xr)
    if np.linalg.norm(vh0) > 1e-8:
        theta_h = float(np.arctan2(vh0[1], vh0[0]))
    else:
        theta_h = float(np.arctan2(xh0[1] - xr[1], xh0[0] - xr[0]))

    z = -0.01
    u = 0.01

    traj = np.zeros((total_steps, 2), dtype=float)

    for step in range(total_steps):
        xh, vh = _human_state_for_opinion(human_positions, human_velocities, step, xr)
        if np.linalg.norm(vh) > 1e-8:
            theta_h = float(np.arctan2(vh[1], vh[0]))

        eta_body = wrap_to_pi(np.arctan2(xr[1] - xh[1], xr[0] - xh[0]) - theta_h)
        chi = float(np.linalg.norm(xh - xr))
        kappa = max(float(np.cos(eta_body)), 0.0)

        desired_angle = wrap_to_pi(np.arctan2(xrg[1] - xr[1], xrg[0] - xr[0]) - theta)
        phi_r = desired_angle
        z_hat = np.sin(-eta_body)

        v_r_vec = robot_speed * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        rel_pos = xh - xr
        rel_vel = vh - v_r_vec
        ttc_proj = _projected_ttc_scalar(rel_pos, rel_vel, collision_distance)
        ttc_input = np.inf if np.isnan(ttc_proj) else ttc_proj

        z_dot = -opinion_params.dr * z + u * np.tanh(opinion_params.alpha_r * z + opinion_params.gamma_r * z_hat)
        if attention_mode == "kappa":
            u_target = attention_dynamics_kappa(opinion_params, chi, kappa)
        elif attention_mode == "ttc":
            u_target = attention_dynamics_ttc(opinion_params, ttc_input)
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
        u_dot = (-u + u_target) / opinion_params.tau_u

        z += z_dot * dt
        u += u_dot * dt
        omega = opinion_params.kr * np.sin(opinion_params.beta_r * np.tanh(z) + phi_r)
        theta = wrap_to_pi(theta + omega * dt)

        if np.linalg.norm(xr - xrg) > goal_tolerance:
            xr = xr + robot_speed * np.array([np.cos(theta), np.sin(theta)], dtype=float) * dt

        traj[step] = xr

    return traj


def evaluate_opinion_dataset(
    *,
    dataset_path: Path,
    output_csv: Path,
    scenario_summary_csv: Path,
    max_samples: int | None,
    max_vel: float,
    attention_mode: str,
    goal_tolerance: float,
    collision_distance: float,
    extra_steps: int,
    opinion_params: OpinionParams,
) -> Dict[str, Any]:
    data = np.load(dataset_path, allow_pickle=True)
    human_positions_all = data["human_positions"]
    human_velocities_all = data["human_velocities"]
    robot_starts_all = data["robot_starts"]
    robot_goals_all = data["robot_goals"]
    scenarios = data["scenarios"]
    speed_scales = data["speed_scales"] if "speed_scales" in data.files else np.ones(human_positions_all.shape[0])
    robot_speeds = data["robot_speeds"] if "robot_speeds" in data.files else None
    dt = float(data["dt"])

    n_total = int(human_positions_all.shape[0])
    n_eval = min(n_total, max_samples) if max_samples is not None else n_total

    rows: List[Dict[str, Any]] = []
    for sample_id in range(n_eval):
        human_positions = human_positions_all[sample_id]
        human_velocities = human_velocities_all[sample_id]
        robot_start = np.asarray(robot_starts_all[sample_id], dtype=float).reshape(2)
        robot_goal = np.asarray(robot_goals_all[sample_id], dtype=float).reshape(2)
        robot_speed = float(robot_speeds[sample_id]) if robot_speeds is not None else float(max_vel)

        robot_positions = _run_opinion_rollout_from_dataset(
            human_positions=human_positions,
            human_velocities=human_velocities,
            robot_start=robot_start,
            robot_goal=robot_goal,
            dt=dt,
            robot_speed=robot_speed,
            opinion_params=opinion_params,
            attention_mode=attention_mode,
            collision_distance=collision_distance,
            goal_tolerance=goal_tolerance,
            extra_steps=extra_steps,
        )

        min_distance_series = _distance_series_to_humans(robot_positions, human_positions)
        min_dist = float(np.min(min_distance_series))
        path_length = _path_length(robot_positions)
        mean_curvature = _mean_curvature(robot_positions)
        projected_ttc = _projected_time_to_collision(
            robot_positions=robot_positions,
            human_positions=human_positions,
            human_velocities=human_velocities,
            collision_distance=collision_distance,
            dt=dt,
        )

        success, final_dist, time_to_goal = _goal_reached(robot_positions, robot_goal, goal_tolerance, dt)
        collision = min_dist < collision_distance
        timeout = (not success) and (not collision)

        rows.append(
            {
                "sample_id": sample_id,
                "scenario": str(scenarios[sample_id]),
                "speed_scale": float(speed_scales[sample_id]),
                "robot_speed": robot_speed,
                "attention_mode": attention_mode,
                "opinion_min_dist": min_dist,
                "opinion_path_length": path_length,
                "opinion_mean_curvature": mean_curvature,
                "opinion_time_to_collision": projected_ttc,
                "opinion_projected_ttc": projected_ttc,
                "opinion_final_dist_to_goal": final_dist,
                "opinion_time_to_goal": time_to_goal,
                "opinion_collision": int(collision),
                "opinion_success": int(success),
                "opinion_timeout": int(timeout),
            }
        )

    if not rows:
        raise ValueError("No samples evaluated. Check --max-samples and dataset content.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scenario_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        scenario_groups.setdefault(row["scenario"], []).append(row)

    scenario_rows = []
    for scenario, group in sorted(scenario_groups.items()):
        coll = np.array([r["opinion_collision"] for r in group], dtype=np.float64)
        succ = np.array([r["opinion_success"] for r in group], dtype=np.float64)
        tout = np.array([r["opinion_timeout"] for r in group], dtype=np.float64)
        min_dist = np.array([r["opinion_min_dist"] for r in group], dtype=np.float64)
        path_len = np.array([r["opinion_path_length"] for r in group], dtype=np.float64)
        curvature = np.array([r["opinion_mean_curvature"] for r in group], dtype=np.float64)
        ttc = np.array([r["opinion_time_to_collision"] for r in group], dtype=np.float64)
        t_goal = np.array([r["opinion_time_to_goal"] for r in group], dtype=np.float64)

        scenario_rows.append(
            {
                "scenario": scenario,
                "n_samples": len(group),
                "opinion_collision_rate": float(coll.mean()),
                "opinion_success_rate": float(succ.mean()),
                "opinion_timeout_rate": float(tout.mean()),
                "opinion_mean_min_dist": float(min_dist.mean()),
                "opinion_mean_path_length": _safe_nanmean(path_len),
                "opinion_mean_curvature": _safe_nanmean(curvature),
                "opinion_mean_time_to_collision": _safe_nanmean(ttc),
                "opinion_mean_projected_ttc": _safe_nanmean(ttc),
                "opinion_mean_time_to_goal": _safe_nanmean(t_goal),
            }
        )

    scenario_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with scenario_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    arr_collision = np.array([r["opinion_collision"] for r in rows], dtype=np.float64)
    arr_success = np.array([r["opinion_success"] for r in rows], dtype=np.float64)
    arr_timeout = np.array([r["opinion_timeout"] for r in rows], dtype=np.float64)
    arr_min_dist = np.array([r["opinion_min_dist"] for r in rows], dtype=np.float64)
    arr_path_len = np.array([r["opinion_path_length"] for r in rows], dtype=np.float64)
    arr_curvature = np.array([r["opinion_mean_curvature"] for r in rows], dtype=np.float64)
    arr_ttc = np.array([r["opinion_time_to_collision"] for r in rows], dtype=np.float64)

    return {
        "n_evaluated": n_eval,
        "used_dataset_robot_speeds": bool(robot_speeds is not None),
        "attention_mode": attention_mode,
        "opinion_collision_rate": float(arr_collision.mean()),
        "opinion_success_rate": float(arr_success.mean()),
        "opinion_timeout_rate": float(arr_timeout.mean()),
        "opinion_mean_min_dist": float(arr_min_dist.mean()),
        "opinion_mean_path_length": _safe_nanmean(arr_path_len),
        "opinion_mean_curvature": _safe_nanmean(arr_curvature),
        "opinion_mean_time_to_collision": _safe_nanmean(arr_ttc),
        "opinion_mean_projected_ttc": _safe_nanmean(arr_ttc),
        "output_csv": str(output_csv),
        "output_scenario_summary_csv": str(scenario_summary_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate opinion-dynamics dataset or evaluate opinion dynamics on an existing .npz dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "evaluate-dataset"],
        default="generate",
        help="generate: legacy .mat generation flow; evaluate-dataset: run opinion rollout on a provided .npz dataset",
    )

    parser.add_argument("--dataset", default="data/human_robot_trajectories_kinematic.npz", help="Input dataset .npz")
    parser.add_argument(
        "--output-csv",
        default="data/opinion_batch_eval.csv",
        help="Output CSV with per-sample opinion metrics (evaluate-dataset mode)",
    )
    parser.add_argument(
        "--scenario-summary-csv",
        default="data/opinion_batch_eval_by_scenario.csv",
        help="Output CSV with aggregated opinion metrics by scenario (evaluate-dataset mode)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only first N samples")
    parser.add_argument("--max-vel", type=float, default=1.0, help="Fallback robot velocity when dataset has no robot_speeds")
    parser.add_argument("--goal-tolerance", type=float, default=0.2, help="Goal success tolerance")
    parser.add_argument("--collision-distance", type=float, default=0.4, help="Collision threshold distance")
    parser.add_argument("--extra-steps", type=int, default=40, help="Extra rollout steps after dataset horizon")
    parser.add_argument("--attention-mode", choices=["kappa", "ttc"], default="ttc")

    parser.add_argument("--dr", type=float, default=2.4)
    parser.add_argument("--alpha-r", type=float, default=0.3)
    parser.add_argument("--gamma-r", type=float, default=10.0)
    parser.add_argument("--Rr", type=float, default=6.0)
    parser.add_argument("--kr", type=float, default=1.5)
    parser.add_argument("--beta-r", type=float, default=float(np.pi / 4.0))
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--tau-u", type=float, default=1.0)
    parser.add_argument(
        "--plot-comparison-samples",
        type=int,
        default=0,
        help="If >0 in generate mode, save this many random trajectory comparison plots",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=0,
        help="Random seed used to choose comparison samples for plotting",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="data/trajectory_plots_comparison",
        help="Output directory for generated trajectory comparison plots",
    )
    parser.add_argument(
        "--animate-oval-sample",
        type=int,
        default=-1,
        help="If >=0 in generate mode, animate this index from available sample keys for traj5+oval",
    )
    parser.add_argument(
        "--animate-oval-output",
        default="data/trajectory_plots_comparison/traj5_oval_animation.gif",
        help="Output GIF path for traj5+oval animation",
    )

    return parser.parse_args()


def run_legacy_generation(args: argparse.Namespace) -> None:
    try:
        from scipy.io import savemat
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Legacy generation mode requires scipy. Install it with: pip install scipy, "
            "or run with --mode evaluate-dataset for dataset-based evaluation."
        ) from exc

    data_folder = os.path.join("..", "opinion_dynamics_dataset_generated")
    if os.path.isdir(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder, exist_ok=True)

    N_per_scenario = 20
    scenarios = ["meeting", "meeting_delayed_arc", "abrupt_change", "step_pattern", "zigzag"]
    attention_modes = ["kappa", "ttc"]
    v_human_list = [0.8, 1.0]
    v_robot_list = [0.5, 0.7]

    dt = 0.01
    max_time = 30.0
    collision_distance = 1.0
    goal_threshold = 0.5

    np.random.seed(1234)
    metrics: List[Dict[str, Any]] = []
    sample_store: Dict[Tuple[int, float, float], Dict[str, Any]] = {}

    human_data = generate_human_paths(N_per_scenario, scenarios)
    savemat(os.path.join(data_folder, "human_paths.mat"), {"human_data": human_data})

    idx = 1
    for s_idx, sample in enumerate(human_data, start=1):
        scenario = sample["scenario"]
        human_path = np.asarray(sample["path"], dtype=float)

        robot_start, robot_goal = make_robot_start_goal(scenario, human_path)
        robot_start = np.asarray(robot_start, dtype=float).reshape(2)
        robot_goal = np.asarray(robot_goal, dtype=float).reshape(2)

        for attention_mode in attention_modes:
            for vh in v_human_list:
                for vr in v_robot_list:
                    np.random.seed(1000 + s_idx)

                    print(
                        f"Sim {idx:04d} | {scenario} | attn={attention_mode} | "
                        f"v_h={vh:.2f} v_r={vr:.2f}"
                    )

                    sim_result = simulate_social_nav(
                        robot_start,
                        robot_goal,
                        human_path,
                        attention_mode,
                        vh,
                        vr,
                        dt,
                        max_time,
                    )

                    sample_key = (s_idx, float(vh), float(vr))
                    if sample_key not in sample_store:
                        sample_store[sample_key] = {
                            "scenario": scenario,
                            "v_human": float(vh),
                            "v_robot": float(vr),
                            "robot_start": robot_start.copy(),
                            "robot_goal": robot_goal.copy(),
                        }
                    sample_store[sample_key]["human_traj"] = sim_result["human_traj"].copy()
                    if attention_mode == "kappa":
                        sample_store[sample_key]["traj1_kappa"] = sim_result["robot_traj"].copy()
                        sample_store[sample_key]["traj5_kappa"] = sim_result["robot_traj5"].copy()
                        sample_store[sample_key]["avocado"] = sim_result["robot_traj_avocado"].copy()
                    elif attention_mode == "ttc":
                        sample_store[sample_key]["traj1_ttc"] = sim_result["robot_traj"].copy()

                    sim_result["meta"] = {
                        "scenario": scenario,
                        "attention_mode": attention_mode,
                        "v_human": vh,
                        "v_robot": vr,
                        "robot_start": robot_start,
                        "robot_goal": robot_goal,
                        "dt": dt,
                        "max_time": max_time,
                    }

                    sim_result["metrics_list"] = []
                    variant_names = ["robot_traj", "robot_traj5", "robot_traj_avocado"]
                    for field in variant_names:
                        if field in sim_result:
                            r_traj = sim_result[field]
                            T = min(r_traj.shape[1], sim_result["human_traj"].shape[1])
                            r_traj = r_traj[:, :T]
                            h_traj = sim_result["human_traj"][:, :T]
                            m = compute_metrics(
                                r_traj,
                                h_traj,
                                dt,
                                scenario,
                                collision_distance,
                            )
                            m["v_human"] = vh
                            m["v_robot"] = vr
                            m["attention_mode"] = attention_mode
                            m["variant"] = field
                            m["goal_distance_final"] = float(np.linalg.norm(r_traj[:, -1] - robot_goal))
                            m["reached_goal"] = bool(m["goal_distance_final"] <= goal_threshold)
                            m["collision"] = bool(m["min_distance"] < collision_distance)
                            m["success"] = bool(m["reached_goal"] and not m["collision"])
                            m["safety_efficiency_tradeoff"] = float(
                                m["min_distance"] / (m["path_length"] + 1e-6)
                            )
                            print(
                                f"  [{field}] TTC(py projected)={m['min_ttc_proj']:.4f} s | "
                                f"radius={m['collision_distance']:.2f} m"
                            )
                            sim_result["metrics_list"].append(m)
                            metrics.append(m)

                    sim_result["metrics"] = (
                        sim_result["metrics_list"][0]
                        if sim_result["metrics_list"]
                        else {}
                    )

                    filename = (
                        f"{idx:04d}_{scenario}_attn-{attention_mode}_vh-{vh:.2f}_vr-{vr:.2f}.mat"
                    )
                    savemat(os.path.join(data_folder, filename), {"sim_result": sim_result})
                    idx += 1

    print(f"Dataset generation complete. Files saved under: {data_folder}")

    if not metrics:
        print("No metrics collected.")
        return

    print("\n=== FOCUSED IMPROVEMENTS (baseline: traj1 + kappa) ===")
    print("traj1 = robot_traj, traj5 = robot_traj5 (ovals), avocado = robot_traj_avocado\n")

    all_success_rate = 100.0 * np.mean([float(m["success"]) for m in metrics])
    all_collision_rate = 100.0 * np.mean([float(m["collision"]) for m in metrics])
    print(f"Overall rates: Success={all_success_rate:.2f}% | Collision={all_collision_rate:.2f}%")

    sel_base = [m for m in metrics if m["variant"] == "robot_traj" and m["attention_mode"] == "kappa"]
    sel_t1_ttc = [m for m in metrics if m["variant"] == "robot_traj" and m["attention_mode"] == "ttc"]
    sel_t5_kappa = [m for m in metrics if m["variant"] == "robot_traj5" and m["attention_mode"] == "kappa"]
    sel_avocado = [m for m in metrics if m["variant"] == "robot_traj_avocado" and m["attention_mode"] == "kappa"]

    rate_sets = [
        ("traj1 + kappa", sel_base),
        ("traj1 + ttc", sel_t1_ttc),
        ("traj5 (ovals) + kappa", sel_t5_kappa),
        ("avocado", sel_avocado),
    ]

    print("Rates by group:")
    for label, group_metrics in rate_sets:
        if group_metrics:
            succ_rate = 100.0 * np.mean([float(m["success"]) for m in group_metrics])
            coll_rate = 100.0 * np.mean([float(m["collision"]) for m in group_metrics])
            print(f"  {label}: Success={succ_rate:.2f}% | Collision={coll_rate:.2f}%")
        else:
            print(f"  {label}: n/a")
    print("")

    if args.plot_comparison_samples > 0:
        complete_keys = [
            k for k, rec in sample_store.items()
            if all(name in rec for name in ("human_traj", "traj1_kappa", "traj1_ttc", "traj5_kappa", "avocado"))
        ]
        if complete_keys:
            n_plot = min(int(args.plot_comparison_samples), len(complete_keys))
            rng = np.random.default_rng(args.plot_seed)
            picked_idx = rng.choice(len(complete_keys), size=n_plot, replace=False)
            picked_keys = [complete_keys[int(i)] for i in picked_idx]
            _plot_legacy_comparison_samples(
                sample_keys=picked_keys,
                sample_store=sample_store,
                output_dir=Path(args.plot_output_dir),
            )
            print(f"Saved {n_plot} comparison plots to: {Path(args.plot_output_dir)}")
        else:
            print("No complete sample groups available for comparison plotting.")

    if args.animate_oval_sample >= 0:
        anim_keys = [
            k for k, rec in sample_store.items()
            if all(name in rec for name in ("human_traj", "traj5_kappa"))
        ]
        if anim_keys:
            anim_keys = sorted(anim_keys)
            pick = int(np.clip(args.animate_oval_sample, 0, len(anim_keys) - 1))
            key = anim_keys[pick]
            rec = sample_store[key]
            _animate_traj5_with_oval(
                human_traj=rec["human_traj"],
                traj5=rec["traj5_kappa"],
                robot_start=rec["robot_start"],
                robot_goal=rec["robot_goal"],
                output_path=Path(args.animate_oval_output),
            )
            print(f"Saved traj5+oval animation to: {Path(args.animate_oval_output)}")
        else:
            print("No eligible sample found for traj5+oval animation.")

    if not sel_base:
        print("Missing baseline (traj1 + kappa).")
        return

    baseline_stats = {
        "path": _mean_metric(sel_base, "path_length"),
        "dist": _mean_metric(sel_base, "min_distance"),
        "smooth": _mean_metric(sel_base, "path_smoothness"),
        "ttc": _mean_metric(sel_base, "min_ttc_proj"),
        "curv": _mean_metric(sel_base, "avg_curvature"),
    }

    print(
        "Baseline values: "
        f"Path {baseline_stats['path']:.4f} | "
        f"MinDist {baseline_stats['dist']:.4f} | "
        f"Smooth {baseline_stats['smooth']:.4f} | "
        f"MinTTCproj {baseline_stats['ttc']:.4f} | "
        f"Curv {baseline_stats['curv']:.4f}"
    )

    if sel_t1_ttc:
        cand = {
            "path": _mean_metric(sel_t1_ttc, "path_length"),
            "dist": _mean_metric(sel_t1_ttc, "min_distance"),
            "smooth": _mean_metric(sel_t1_ttc, "path_smoothness"),
            "ttc": _mean_metric(sel_t1_ttc, "min_ttc_proj"),
            "curv": _mean_metric(sel_t1_ttc, "avg_curvature"),
        }
        _print_improvement("Improvement A: traj1 + ttc vs traj1 + kappa", baseline_stats, cand)
    else:
        print("Improvement A: traj1 + ttc data not available.")

    if sel_t5_kappa:
        cand = {
            "path": _mean_metric(sel_t5_kappa, "path_length"),
            "dist": _mean_metric(sel_t5_kappa, "min_distance"),
            "smooth": _mean_metric(sel_t5_kappa, "path_smoothness"),
            "ttc": _mean_metric(sel_t5_kappa, "min_ttc_proj"),
            "curv": _mean_metric(sel_t5_kappa, "avg_curvature"),
        }
        _print_improvement("Improvement B: traj5 (ovals) + kappa vs traj1 + kappa", baseline_stats, cand)
    else:
        print("Improvement B: traj5 (ovals) + kappa data not available.")

    if sel_avocado:
        cand = {
            "path": _mean_metric(sel_avocado, "path_length"),
            "dist": _mean_metric(sel_avocado, "min_distance"),
            "smooth": _mean_metric(sel_avocado, "path_smoothness"),
            "ttc": _mean_metric(sel_avocado, "min_ttc_proj"),
            "curv": _mean_metric(sel_avocado, "avg_curvature"),
        }
        _print_improvement("Improvement C: AVOCADO vs traj1 + kappa", baseline_stats, cand)
    else:
        print("Improvement C: AVOCADO data not available.")

    print("")


def main() -> None:
    args = parse_args()

    if args.mode == "generate":
        run_legacy_generation(args)
        return

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    opinion_params = get_opinion_params(
        {
            "dr": args.dr,
            "alpha_r": args.alpha_r,
            "gamma_r": args.gamma_r,
            "Rr": args.Rr,
            "kr": args.kr,
            "beta_r": args.beta_r,
            "u_max": args.u_max,
            "u_min": args.u_min,
            "n": args.n,
            "tau_u": args.tau_u,
        }
    )

    summary = evaluate_opinion_dataset(
        dataset_path=dataset_path,
        output_csv=Path(args.output_csv),
        scenario_summary_csv=Path(args.scenario_summary_csv),
        max_samples=args.max_samples,
        max_vel=args.max_vel,
        attention_mode=args.attention_mode,
        goal_tolerance=args.goal_tolerance,
        collision_distance=args.collision_distance,
        extra_steps=args.extra_steps,
        opinion_params=opinion_params,
    )

    print("Opinion-dynamics batch evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
