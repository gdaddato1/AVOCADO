import argparse
import csv
from pathlib import Path

import numpy as np

from actors import AVOCADO_Actor


def _ensure_human_shape(arr_t):
    if arr_t.ndim == 1:
        return arr_t.reshape(1, 2)
    if arr_t.ndim == 2:
        return arr_t
    raise ValueError("Unexpected human array shape. Expected (2,) or (H,2) per frame.")


def _run_avocado_rollout(
    human_positions,
    human_velocities,
    robot_start,
    robot_goal,
    dt,
    max_vel,
    actor_params,
    extra_steps=0,
):
    actor = AVOCADO_Actor(
        agent_radius=actor_params["agent_radius"],
        timestep=dt,
        alpha=[actor_params["alpha"]],
        a=actor_params["a"],
        c=actor_params["c"],
        d=actor_params["d"],
        kappa=actor_params["kappa"],
        epsilon=actor_params["epsilon"],
        delta=actor_params["delta"],
        bias=[actor_params["bias"]],
    )

    n_steps = human_positions.shape[0]
    total_steps = n_steps + max(0, int(extra_steps))
    robot_pos = np.array(robot_start, dtype=np.float64)
    robot_goal = np.array(robot_goal, dtype=np.float64)
    robot_vel = np.zeros(2, dtype=np.float64)

    avocado_positions = np.zeros((total_steps, 2), dtype=np.float64)

    for t in range(total_steps):
        if t < n_steps:
            other_pos_t = _ensure_human_shape(human_positions[t])
            other_vel_t = _ensure_human_shape(human_velocities[t])
        else:
            other_pos_t = _ensure_human_shape(human_positions[-1])
            other_vel_t = np.zeros_like(other_pos_t)

        cmd = actor.act(
            agent_positions=robot_pos.reshape(1, 2),
            other_positions=other_pos_t,
            agent_velocities=robot_vel.reshape(1, 2),
            other_velocities=other_vel_t,
            agent_goals=robot_goal.reshape(1, 2),
            max_vel=max_vel,
        )[0]

        robot_vel = cmd
        robot_pos = robot_pos + robot_vel * dt
        avocado_positions[t] = robot_pos

    return avocado_positions


def _distance_series_to_humans(robot_positions, human_positions):
    if human_positions.ndim == 2:
        n_h = human_positions.shape[0]
        n_r = robot_positions.shape[0]
        n_common = min(n_h, n_r)
        distances_common = np.linalg.norm(robot_positions[:n_common] - human_positions[:n_common], axis=1)
        if n_r > n_h:
            last_h = human_positions[-1]
            distances_extra = np.linalg.norm(robot_positions[n_h:] - last_h, axis=1)
            distances = np.concatenate((distances_common, distances_extra))
        else:
            distances = distances_common
        return distances
    if human_positions.ndim == 3:
        n_h = human_positions.shape[0]
        n_r = robot_positions.shape[0]
        n_common = min(n_h, n_r)
        distances_common = np.linalg.norm(robot_positions[:n_common, None, :] - human_positions[:n_common], axis=2).min(axis=1)
        if n_r > n_h:
            last_h = human_positions[-1]
            distances_extra = np.linalg.norm(robot_positions[n_h:, None, :] - last_h[None, :, :], axis=2).min(axis=1)
            distances = np.concatenate((distances_common, distances_extra))
        else:
            distances = distances_common
        return distances
    raise ValueError("Unexpected human_positions shape")


def _path_length(robot_positions):
    if robot_positions.shape[0] < 2:
        return 0.0
    segments = np.diff(robot_positions, axis=0)
    return float(np.linalg.norm(segments, axis=1).sum())


def _velocities(positions, dt):
    vel = np.zeros_like(positions)
    if positions.shape[0] > 1:
        vel[1:] = (positions[1:] - positions[:-1]) / dt
        vel[0] = vel[1]
    return vel


def _mean_curvature(robot_positions):
    if robot_positions.shape[0] < 3:
        return np.nan

    dpos = np.diff(robot_positions, axis=0)
    ds = np.linalg.norm(dpos, axis=1)
    valid = ds > 1e-8
    if np.count_nonzero(valid) < 2:
        return np.nan

    headings = np.arctan2(dpos[:, 1], dpos[:, 0])
    dtheta = np.diff(headings)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    ds_mid = 0.5 * (ds[:-1] + ds[1:])
    valid_kappa = ds_mid > 1e-8
    if not np.any(valid_kappa):
        return np.nan

    kappa = np.abs(dtheta[valid_kappa]) / ds_mid[valid_kappa]
    return float(np.mean(kappa)) if kappa.size > 0 else np.nan


def _time_to_collision(min_distance_series, collision_distance, dt):
    idx = np.where(min_distance_series < collision_distance)[0]
    if idx.size == 0:
        return np.nan
    return float(idx[0] * dt)


def _pairwise_projected_ttc(rel_pos, rel_vel, collision_distance):
    c = float(np.dot(rel_pos, rel_pos) - collision_distance**2)
    if c <= 0.0:
        return 0.0

    a = float(np.dot(rel_vel, rel_vel))
    b = 2.0 * float(np.dot(rel_pos, rel_vel))

    if a < 1e-12:
        return np.nan

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return np.nan

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return np.nan
    return float(min(candidates))


def _human_state_at_step(human_positions, human_velocities, step):
    if human_positions.ndim == 2:
        idx = min(step, human_positions.shape[0] - 1)
        pos = _ensure_human_shape(human_positions[idx])
        if step < human_velocities.shape[0]:
            vel = _ensure_human_shape(human_velocities[idx])
        else:
            vel = np.zeros_like(pos)
        return pos, vel

    if human_positions.ndim == 3:
        if step < human_positions.shape[0]:
            pos = human_positions[step]
            vel = human_velocities[step]
        else:
            pos = human_positions[-1]
            vel = np.zeros_like(pos)
        return pos, vel

    raise ValueError("Unexpected human_positions shape")


def _projected_time_to_collision(robot_positions, human_positions, human_velocities, collision_distance, dt):
    robot_velocities = _velocities(robot_positions, dt)
    best_abs_ttc = np.inf

    for step in range(robot_positions.shape[0]):
        robot_pos = robot_positions[step]
        robot_vel = robot_velocities[step]
        human_pos_t, human_vel_t = _human_state_at_step(human_positions, human_velocities, step)

        rel_pos = human_pos_t - robot_pos[None, :]
        rel_vel = human_vel_t - robot_vel[None, :]

        for rel_pos_i, rel_vel_i in zip(rel_pos, rel_vel):
            ttc_now = _pairwise_projected_ttc(rel_pos_i, rel_vel_i, collision_distance)
            if np.isnan(ttc_now):
                continue
            abs_ttc = step * dt + ttc_now
            if abs_ttc < best_abs_ttc:
                best_abs_ttc = abs_ttc

    return float(best_abs_ttc) if np.isfinite(best_abs_ttc) else np.nan


def _safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _goal_reached(robot_positions, robot_goal, goal_tolerance, dt):
    distances = np.linalg.norm(robot_positions - robot_goal, axis=1)
    reached_idx = np.where(distances <= goal_tolerance)[0]
    final_distance = float(distances[-1])
    if reached_idx.size > 0:
        return True, final_distance, float(reached_idx[0] * dt)
    return False, final_distance, np.nan


def _plot_random_trajectory_samples(
    *,
    sample_ids,
    human_positions_all,
    robot_starts_all,
    robot_goals_all,
    robot_rollouts_by_sample,
    scenarios,
    output_dir,
):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in sample_ids:
        human_positions = human_positions_all[sample_id]
        robot_traj = robot_rollouts_by_sample[sample_id]
        robot_start = robot_starts_all[sample_id]
        robot_goal = robot_goals_all[sample_id]
        scenario = str(scenarios[sample_id])

        fig, ax = plt.subplots(figsize=(6, 6))

        if human_positions.ndim == 2:
            ax.plot(human_positions[:, 0], human_positions[:, 1], color="tab:blue", linewidth=1.8, label="human")
            ax.scatter(human_positions[0, 0], human_positions[0, 1], color="tab:blue", marker="o", s=30)
            ax.scatter(human_positions[-1, 0], human_positions[-1, 1], color="tab:blue", marker="x", s=36)
        elif human_positions.ndim == 3:
            for idx in range(human_positions.shape[1]):
                traj = human_positions[:, idx, :]
                ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.0, alpha=0.6)

        ax.plot(robot_traj[:, 0], robot_traj[:, 1], color="tab:orange", linewidth=2.2, label="avocado")
        ax.scatter(robot_start[0], robot_start[1], color="tab:orange", marker="^", s=55, label="robot start")
        ax.scatter(robot_goal[0], robot_goal[1], color="tab:green", marker="*", s=90, label="robot goal")

        ax.set_title(f"sample {sample_id} | scenario: {scenario}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        output_path = output_dir / f"trajectory_sample_{sample_id}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def evaluate_dataset(
    dataset_path,
    output_csv,
    scenario_summary_csv,
    max_vel,
    actor_params,
    goal_tolerance,
    collision_distance,
    max_samples,
    extra_steps,
    plot_random_samples,
    plot_seed,
    plot_output_dir,
):
    data = np.load(dataset_path, allow_pickle=True)

    dataset_stem = dataset_path.stem.lower()
    looks_smooth_by_name = ("kinematic" in dataset_stem) or ("smooth" in dataset_stem)
    if not looks_smooth_by_name:
        raise ValueError(
            "This evaluator is configured to use only smooth human trajectories. "
            "Please pass a smooth dataset (e.g., data/human_robot_trajectories_kinematic.npz)."
        )

    human_positions_all = data["human_positions"]
    human_velocities_all = data["human_velocities"]
    robot_starts_all = data["robot_starts"]
    robot_goals_all = data["robot_goals"]
    scenarios = data["scenarios"]
    speed_scales = data["speed_scales"]
    robot_speeds = data["robot_speeds"] if "robot_speeds" in data.files else None
    dt = float(data["dt"])

    n_total = human_positions_all.shape[0]
    n_eval = min(n_total, max_samples) if max_samples is not None else n_total

    rows = []
    robot_rollouts_by_sample = {}

    for sample_id in range(n_eval):
        human_positions = human_positions_all[sample_id]
        human_velocities = human_velocities_all[sample_id]
        robot_start = robot_starts_all[sample_id]
        robot_goal = robot_goals_all[sample_id]
        sample_robot_speed = float(robot_speeds[sample_id]) if robot_speeds is not None else float(max_vel)

        robot_avocado_positions = _run_avocado_rollout(
            human_positions=human_positions,
            human_velocities=human_velocities,
            robot_start=robot_start,
            robot_goal=robot_goal,
            dt=dt,
            max_vel=sample_robot_speed,
            actor_params=actor_params,
            extra_steps=extra_steps,
        )
        robot_rollouts_by_sample[sample_id] = robot_avocado_positions

        min_distance_series = _distance_series_to_humans(robot_avocado_positions, human_positions)
        min_dist_avocado = float(np.min(min_distance_series))
        avocado_path_length = _path_length(robot_avocado_positions)
        avocado_mean_curvature = _mean_curvature(robot_avocado_positions)
        avocado_projected_ttc = _projected_time_to_collision(
            robot_positions=robot_avocado_positions,
            human_positions=human_positions,
            human_velocities=human_velocities,
            collision_distance=collision_distance,
            dt=dt,
        )
        avocado_time_to_collision = avocado_projected_ttc

        avocado_success, avocado_final_dist, avocado_time_to_goal = _goal_reached(
            robot_avocado_positions, robot_goal, goal_tolerance, dt
        )

        avocado_collision = min_dist_avocado < collision_distance

        avocado_timeout = (not avocado_success) and (not avocado_collision)

        rows.append(
            {
                "sample_id": sample_id,
                "scenario": str(scenarios[sample_id]),
                "speed_scale": float(speed_scales[sample_id]),
                "robot_speed": sample_robot_speed,
                "avocado_min_dist": min_dist_avocado,
                "avocado_path_length": avocado_path_length,
                "avocado_mean_curvature": avocado_mean_curvature,
                "avocado_time_to_collision": avocado_time_to_collision,
                "avocado_projected_ttc": avocado_projected_ttc,
                "avocado_final_dist_to_goal": avocado_final_dist,
                "avocado_time_to_goal": avocado_time_to_goal,
                "avocado_collision": int(avocado_collision),
                "avocado_success": int(avocado_success),
                "avocado_timeout": int(avocado_timeout),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scenario_groups = {}
    for row in rows:
        scenario = row["scenario"]
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(row)

    scenario_rows = []
    for scenario, group in sorted(scenario_groups.items()):
        coll = np.array([r["avocado_collision"] for r in group], dtype=np.float64)
        succ = np.array([r["avocado_success"] for r in group], dtype=np.float64)
        tout = np.array([r["avocado_timeout"] for r in group], dtype=np.float64)
        min_dist = np.array([r["avocado_min_dist"] for r in group], dtype=np.float64)
        path_len = np.array([r["avocado_path_length"] for r in group], dtype=np.float64)
        curvature = np.array([r["avocado_mean_curvature"] for r in group], dtype=np.float64)
        ttc = np.array([r["avocado_time_to_collision"] for r in group], dtype=np.float64)
        projected_ttc = np.array([r["avocado_projected_ttc"] for r in group], dtype=np.float64)
        t_goal = np.array([r["avocado_time_to_goal"] for r in group], dtype=np.float64)

        scenario_rows.append(
            {
                "scenario": scenario,
                "n_samples": len(group),
                "avocado_collision_rate": float(coll.mean()),
                "avocado_success_rate": float(succ.mean()),
                "avocado_timeout_rate": float(tout.mean()),
                "avocado_mean_min_dist": float(min_dist.mean()),
                "avocado_mean_path_length": _safe_nanmean(path_len),
                "avocado_mean_curvature": _safe_nanmean(curvature),
                "avocado_mean_time_to_collision": _safe_nanmean(ttc),
                "avocado_mean_projected_ttc": _safe_nanmean(projected_ttc),
                "avocado_mean_time_to_goal": _safe_nanmean(t_goal),
            }
        )

    scenario_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with scenario_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    arr_avocado_collision = np.array([r["avocado_collision"] for r in rows], dtype=np.float64)
    arr_avocado_success = np.array([r["avocado_success"] for r in rows], dtype=np.float64)
    arr_avocado_timeout = np.array([r["avocado_timeout"] for r in rows], dtype=np.float64)
    arr_avocado_min_dist = np.array([r["avocado_min_dist"] for r in rows], dtype=np.float64)
    arr_avocado_path_len = np.array([r["avocado_path_length"] for r in rows], dtype=np.float64)
    arr_avocado_curvature = np.array([r["avocado_mean_curvature"] for r in rows], dtype=np.float64)
    arr_avocado_ttc = np.array([r["avocado_time_to_collision"] for r in rows], dtype=np.float64)
    arr_avocado_projected_ttc = np.array([r["avocado_projected_ttc"] for r in rows], dtype=np.float64)

    summary = {
        "n_evaluated": n_eval,
        "used_dataset_robot_speeds": bool(robot_speeds is not None),
        "avocado_collision_rate": float(arr_avocado_collision.mean()),
        "avocado_success_rate": float(arr_avocado_success.mean()),
        "avocado_timeout_rate": float(arr_avocado_timeout.mean()),
        "avocado_mean_min_dist": float(arr_avocado_min_dist.mean()),
        "avocado_mean_path_length": _safe_nanmean(arr_avocado_path_len),
        "avocado_mean_curvature": _safe_nanmean(arr_avocado_curvature),
        "avocado_mean_time_to_collision": _safe_nanmean(arr_avocado_ttc),
        "avocado_mean_projected_ttc": _safe_nanmean(arr_avocado_projected_ttc),
        "output_csv": str(output_csv),
        "output_scenario_summary_csv": str(scenario_summary_csv),
    }

    if plot_random_samples and n_eval > 0:
        n_plot = min(int(plot_random_samples), n_eval)
        rng = np.random.default_rng(plot_seed)
        selected_ids = sorted(rng.choice(n_eval, size=n_plot, replace=False).tolist())
        _plot_random_trajectory_samples(
            sample_ids=selected_ids,
            human_positions_all=human_positions_all,
            robot_starts_all=robot_starts_all,
            robot_goals_all=robot_goals_all,
            robot_rollouts_by_sample=robot_rollouts_by_sample,
            scenarios=scenarios,
            output_dir=plot_output_dir,
        )
        summary["plotted_random_samples"] = selected_ids
        summary["plot_output_dir"] = str(plot_output_dir)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate AVOCADO on precomputed human trajectory dataset.")
    parser.add_argument(
        "--dataset",
        default="data/human_robot_trajectories_kinematic.npz",
        help="Input smooth human trajectory dataset .npz",
    )
    parser.add_argument("--output-csv", default="data/avocado_batch_eval.csv", help="Output CSV with per-sample metrics")
    parser.add_argument(
        "--scenario-summary-csv",
        default="data/avocado_batch_eval_by_scenario.csv",
        help="Output CSV with aggregated metrics by scenario",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only first N samples")
    parser.add_argument("--max-vel", type=float, default=1.0, help="Robot max velocity")
    parser.add_argument("--agent-radius", type=float, default=0.2, help="Robot radius")
    parser.add_argument("--goal-tolerance", type=float, default=0.2, help="Goal success tolerance")
    parser.add_argument("--collision-distance", type=float, default=0.4, help="Collision threshold distance")
    parser.add_argument("--extra-steps", type=int, default=40, help="Extra rollout steps after dataset horizon")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--c", type=float, default=0.7)
    parser.add_argument("--d", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=14.15)
    parser.add_argument("--epsilon", type=float, default=3.22)
    parser.add_argument("--delta", type=float, default=0.57)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument(
        "--plot-random-samples",
        type=int,
        default=0,
        help="If >0, save this many random evaluated trajectory plots as PNG files",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=0,
        help="Random seed used to choose which evaluated samples are plotted",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="data/trajectory_plots",
        help="Directory where random trajectory plots are saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_csv = Path(args.output_csv)
    scenario_summary_csv = Path(args.scenario_summary_csv)
    plot_output_dir = Path(args.plot_output_dir)
    actor_params = {
        "agent_radius": args.agent_radius,
        "alpha": args.alpha,
        "a": args.a,
        "c": args.c,
        "d": args.d,
        "kappa": args.kappa,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "bias": args.bias,
    }

    summary = evaluate_dataset(
        dataset_path=dataset_path,
        output_csv=output_csv,
        scenario_summary_csv=scenario_summary_csv,
        max_vel=args.max_vel,
        actor_params=actor_params,
        goal_tolerance=args.goal_tolerance,
        collision_distance=args.collision_distance,
        max_samples=args.max_samples,
        extra_steps=args.extra_steps,
        plot_random_samples=args.plot_random_samples,
        plot_seed=args.plot_seed,
        plot_output_dir=plot_output_dir,
    )

    print("Batch evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
