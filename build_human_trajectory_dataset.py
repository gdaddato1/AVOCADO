import argparse
import csv
from pathlib import Path

import numpy as np

from generate_human_paths import generate_human_paths
from make_robot_start_goal import make_robot_start_goal


DEFAULT_SCENARIOS = [
    "meeting",
    "meeting_delayed_arc",
    "abrupt_change",
    "step_pattern",
    "zigzag",
]


def _smooth_interpolate_waypoints(points, dst_u):
    n_points = points.shape[0]
    if n_points < 3:
        src_u = np.linspace(0.0, 1.0, n_points)
        x = np.interp(dst_u, src_u, points[:, 0])
        y = np.interp(dst_u, src_u, points[:, 1])
        return np.stack((x, y), axis=1)

    src_u = np.linspace(0.0, 1.0, n_points)
    tangents = np.zeros_like(points, dtype=np.float64)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = 0.5 * (points[2:] - points[:-2])

    values = np.zeros((len(dst_u), 2), dtype=np.float64)
    for idx, u in enumerate(dst_u):
        if u >= 1.0:
            values[idx] = points[-1]
            continue

        seg = np.searchsorted(src_u, u, side="right") - 1
        seg = max(0, min(seg, n_points - 2))

        u0 = src_u[seg]
        u1 = src_u[seg + 1]
        h = u1 - u0
        s = (u - u0) / h

        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        p0 = points[seg]
        p1 = points[seg + 1]
        m0 = tangents[seg]
        m1 = tangents[seg + 1]

        values[idx] = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1

    return values


def _resample_polyline(path_xy, num_steps, speed_scale=1.0):
    points = path_xy.T
    dst_u = np.linspace(0.0, 1.0, num_steps) * speed_scale
    dst_u = np.clip(dst_u, 0.0, 1.0)
    return _smooth_interpolate_waypoints(points, dst_u)


def _velocities(positions, dt):
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) / dt
    vel[0] = vel[1]
    return vel


def build_dataset(
    output_path,
    n_per_scenario,
    scenarios,
    speed_scales,
    robot_speeds,
    n_waypoints,
    num_steps,
    dt,
    seed,
):
    rng = np.random.default_rng(seed)
    human_specs = generate_human_paths(
        n_per_scenario=n_per_scenario,
        scenarios=scenarios,
        n_waypoints=n_waypoints,
        rng=rng,
    )

    human_positions = []
    human_velocities = []
    robot_starts = []
    robot_goals = []
    scenario_labels = []
    speed_labels = []
    robot_speed_labels = []

    for spec in human_specs:
        scenario = spec["scenario"]
        path = spec["path"]
        robot_start, robot_goal = make_robot_start_goal(scenario, path, rng=rng)

        for speed in speed_scales:
            h_pos = _resample_polyline(path, num_steps=num_steps, speed_scale=speed)
            h_vel = _velocities(h_pos, dt=dt)

            for robot_speed in robot_speeds:
                human_positions.append(h_pos.astype(np.float32))
                human_velocities.append(h_vel.astype(np.float32))
                robot_starts.append(robot_start.astype(np.float32))
                robot_goals.append(robot_goal.astype(np.float32))
                scenario_labels.append(scenario)
                speed_labels.append(float(speed))
                robot_speed_labels.append(float(robot_speed))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        human_positions=np.stack(human_positions, axis=0),
        human_velocities=np.stack(human_velocities, axis=0),
        robot_starts=np.stack(robot_starts, axis=0),
        robot_goals=np.stack(robot_goals, axis=0),
        scenarios=np.array(scenario_labels, dtype=object),
        speed_scales=np.array(speed_labels, dtype=np.float32),
        robot_speeds=np.array(robot_speed_labels, dtype=np.float32),
        dt=np.float32(dt),
    )

    index_path = output_path.with_suffix(".csv")
    with index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "scenario", "speed_scale", "robot_speed"])
        for idx, (scenario, speed, robot_speed) in enumerate(
            zip(scenario_labels, speed_labels, robot_speed_labels)
        ):
            writer.writerow([idx, scenario, speed, robot_speed])

    return {
        "n_samples": len(scenario_labels),
        "n_scenarios": len(scenarios),
        "n_per_scenario": n_per_scenario,
        "n_speed_scales": len(speed_scales),
        "n_robot_speeds": len(robot_speeds),
        "output_npz": str(output_path),
        "output_index_csv": str(index_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build human trajectory dataset with scenario and speed variations.")
    parser.add_argument("--output", default="data/human_robot_trajectories.npz", help="Output .npz path")
    parser.add_argument("--n-per-scenario", type=int, default=60, help="Base trajectories per scenario")
    parser.add_argument("--n-waypoints", type=int, default=8, help="Waypoints per base path")
    parser.add_argument("--num-steps", type=int, default=80, help="Timesteps per trajectory")
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep duration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=DEFAULT_SCENARIOS,
        help="Scenario names",
    )
    parser.add_argument(
        "--speed-scales",
        "--human-speeds",
        dest="speed_scales",
        nargs="+",
        type=float,
        default=[0.6, 0.85, 1.0, 1.2, 1.5],
        help="Speed multipliers applied to trajectory progression",
    )
    parser.add_argument(
        "--robot-speeds",
        nargs="+",
        type=float,
        default=[1.0],
        help="Robot speed labels stored per sample",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    info = build_dataset(
        output_path=args.output,
        n_per_scenario=args.n_per_scenario,
        scenarios=args.scenarios,
        speed_scales=args.speed_scales,
        robot_speeds=args.robot_speeds,
        n_waypoints=args.n_waypoints,
        num_steps=args.num_steps,
        dt=args.dt,
        seed=args.seed,
    )
    print("Dataset generated:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
