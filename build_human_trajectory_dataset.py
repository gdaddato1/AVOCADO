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


def _wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _simulate_waypoint_follower(
    path_xy,
    num_steps,
    dt,
    speed_scale=1.0,
    switch_radius=0.2,
    blend_radius=0.8,
    heading_gain=2.0,
    omega_max=1.2,
    turn_slowdown=0.6,
):
    points = path_xy.T.astype(np.float64)
    n_waypoints = points.shape[0]

    if n_waypoints == 0:
        return np.zeros((num_steps, 2), dtype=np.float64)
    if n_waypoints == 1:
        return np.repeat(points, repeats=num_steps, axis=0)

    diffs = np.diff(points, axis=0)
    total_length = float(np.linalg.norm(diffs, axis=1).sum())
    base_speed = (total_length / max((num_steps - 1) * dt, 1e-8)) * float(speed_scale)

    xh = points[0].copy()
    initial_heading = points[1] - points[0]
    theta_h = float(np.arctan2(initial_heading[1], initial_heading[0]))
    current_wp = 1

    traj = np.zeros((num_steps, 2), dtype=np.float64)
    traj[0] = xh

    for t in range(1, num_steps):
        wp_current = points[current_wp]
        if np.linalg.norm(xh - wp_current) < switch_radius and current_wp < (n_waypoints - 1):
            current_wp += 1
            wp_current = points[current_wp]

        wp_next = points[min(current_wp + 1, n_waypoints - 1)]
        d_current = np.linalg.norm(xh - wp_current)
        blend = np.exp(-((d_current / max(blend_radius, 1e-8)) ** 2))
        target = (1.0 - blend) * wp_current + blend * wp_next

        th_des = np.arctan2(target[1] - xh[1], target[0] - xh[0])
        e_th = _wrap_to_pi(th_des - theta_h)
        omega = np.clip(heading_gain * e_th, -omega_max, omega_max)
        theta_h = _wrap_to_pi(theta_h + omega * dt)

        speed_cmd = base_speed / (1.0 + turn_slowdown * abs(omega))
        xh = xh + speed_cmd * np.array([np.cos(theta_h), np.sin(theta_h)]) * dt

        if current_wp == (n_waypoints - 1) and np.linalg.norm(xh - points[-1]) < switch_radius:
            xh = points[-1].copy()

        traj[t] = xh

    return traj


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
    trajectory_model,
    switch_radius,
    blend_radius,
    heading_gain,
    omega_max,
    turn_slowdown,
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
            if trajectory_model == "kinematic":
                h_pos = _simulate_waypoint_follower(
                    path,
                    num_steps=num_steps,
                    dt=dt,
                    speed_scale=speed,
                    switch_radius=switch_radius,
                    blend_radius=blend_radius,
                    heading_gain=heading_gain,
                    omega_max=omega_max,
                    turn_slowdown=turn_slowdown,
                )
            else:
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
        "--trajectory-model",
        choices=["interpolated", "kinematic"],
        default="kinematic",
        help="Trajectory generation model from waypoints",
    )
    parser.add_argument("--switch-radius", type=float, default=0.2, help="Waypoint switching distance for kinematic model")
    parser.add_argument("--blend-radius", type=float, default=0.8, help="Waypoint blend distance for kinematic model")
    parser.add_argument("--heading-gain", type=float, default=2.0, help="Heading controller gain for kinematic model")
    parser.add_argument("--omega-max", type=float, default=1.2, help="Max angular speed [rad/s] for kinematic model")
    parser.add_argument("--turn-slowdown", type=float, default=0.6, help="Turn-dependent slowdown factor for kinematic model")
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
        trajectory_model=args.trajectory_model,
        switch_radius=args.switch_radius,
        blend_radius=args.blend_radius,
        heading_gain=args.heading_gain,
        omega_max=args.omega_max,
        turn_slowdown=args.turn_slowdown,
    )
    print("Dataset generated:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
