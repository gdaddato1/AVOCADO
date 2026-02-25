import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
    robot_pos = np.array(robot_start, dtype=np.float64)
    robot_goal = np.array(robot_goal, dtype=np.float64)
    robot_vel = np.zeros(2, dtype=np.float64)

    avocado_positions = np.zeros((n_steps, 2), dtype=np.float64)
    avocado_velocities = np.zeros((n_steps, 2), dtype=np.float64)

    for t in range(n_steps):
        other_pos_t = _ensure_human_shape(human_positions[t])
        other_vel_t = _ensure_human_shape(human_velocities[t])

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
        avocado_velocities[t] = robot_vel

    return avocado_positions, avocado_velocities


def _compute_min_dist(robot_positions, human_positions):
    if human_positions.ndim == 2:
        d = np.linalg.norm(robot_positions - human_positions, axis=1)
    elif human_positions.ndim == 3:
        d = np.linalg.norm(robot_positions[:, None, :] - human_positions, axis=2)
        d = d.min(axis=1)
    else:
        raise ValueError("Unexpected human_positions shape")
    return float(d.min())


def _plot_static(
    human_positions,
    robot_avocado_positions,
    robot_start,
    robot_goal,
    title,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    if human_positions.ndim == 2:
        ax.plot(human_positions[:, 0], human_positions[:, 1], "r-", lw=2, label="Human")
    else:
        for k in range(human_positions.shape[1]):
            label = "Human" if k == 0 else None
            ax.plot(human_positions[:, k, 0], human_positions[:, k, 1], "r-", lw=1.5, alpha=0.8, label=label)

    ax.plot([robot_start[0], robot_goal[0]], [robot_start[1], robot_goal[1]], "k--", lw=1.5, label="Start→Goal")
    ax.plot(robot_avocado_positions[:, 0], robot_avocado_positions[:, 1], "b-", lw=2, label="Robot AVOCADO")

    ax.scatter(robot_start[0], robot_start[1], c="k", marker="o", s=45)
    ax.scatter(robot_goal[0], robot_goal[1], c="k", marker="x", s=65)
    ax.scatter(robot_avocado_positions[0, 0], robot_avocado_positions[0, 1], c="b", marker="o", s=45)
    ax.scatter(robot_avocado_positions[-1, 0], robot_avocado_positions[-1, 1], c="b", marker="x", s=65)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def _animate(
    human_positions,
    robot_avocado_positions,
    robot_start,
    robot_goal,
    title,
    dt,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    all_x = [np.array([robot_start[0], robot_goal[0]]), robot_avocado_positions[:, 0]]
    all_y = [np.array([robot_start[1], robot_goal[1]]), robot_avocado_positions[:, 1]]

    if human_positions.ndim == 2:
        all_x.append(human_positions[:, 0])
        all_y.append(human_positions[:, 1])
    else:
        for k in range(human_positions.shape[1]):
            all_x.append(human_positions[:, k, 0])
            all_y.append(human_positions[:, k, 1])

    min_x = min(np.min(arr) for arr in all_x) - 0.5
    max_x = max(np.max(arr) for arr in all_x) + 0.5
    min_y = min(np.min(arr) for arr in all_y) - 0.5
    max_y = max(np.max(arr) for arr in all_y) + 0.5
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    if human_positions.ndim == 2:
        human_line, = ax.plot([], [], "r-", lw=2, label="Human")
        human_dot, = ax.plot([], [], "ro", ms=5)
        human_lines = [human_line]
        human_dots = [human_dot]
    else:
        human_lines = []
        human_dots = []
        for k in range(human_positions.shape[1]):
            label = "Human" if k == 0 else None
            line, = ax.plot([], [], "r-", lw=1.5, alpha=0.8, label=label)
            dot, = ax.plot([], [], "ro", ms=4)
            human_lines.append(line)
            human_dots.append(dot)

    robot_ref_line, = ax.plot([robot_start[0], robot_goal[0]], [robot_start[1], robot_goal[1]], "k--", lw=1.5, label="Start→Goal")
    robot_av_line, = ax.plot([], [], "b-", lw=2, label="Robot AVOCADO")
    robot_av_dot, = ax.plot([], [], "bo", ms=5)

    ax.legend(loc="best")

    n_steps = robot_avocado_positions.shape[0]

    def init():
        artists = []
        for line, dot in zip(human_lines, human_dots):
            line.set_data([], [])
            dot.set_data([], [])
            artists.extend([line, dot])
        robot_av_line.set_data([], [])
        robot_av_dot.set_data([], [])
        artists.extend([robot_ref_line, robot_av_line, robot_av_dot])
        return artists

    def update(frame):
        artists = []
        if human_positions.ndim == 2:
            human_lines[0].set_data(human_positions[: frame + 1, 0], human_positions[: frame + 1, 1])
            human_dots[0].set_data(human_positions[frame, 0], human_positions[frame, 1])
            artists.extend([human_lines[0], human_dots[0]])
        else:
            for k, (line, dot) in enumerate(zip(human_lines, human_dots)):
                line.set_data(human_positions[: frame + 1, k, 0], human_positions[: frame + 1, k, 1])
                dot.set_data(human_positions[frame, k, 0], human_positions[frame, k, 1])
                artists.extend([line, dot])

        robot_av_line.set_data(robot_avocado_positions[: frame + 1, 0], robot_avocado_positions[: frame + 1, 1])
        robot_av_dot.set_data(robot_avocado_positions[frame, 0], robot_avocado_positions[frame, 1])
        artists.extend([robot_ref_line, robot_av_line, robot_av_dot])

        return artists

    animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_steps,
        interval=dt * 1000,
        blit=True,
        repeat=False,
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Replay one dataset sample through AVOCADO actor.")
    parser.add_argument("--dataset", default="data/human_robot_trajectories.npz", help="Path to dataset .npz")
    parser.add_argument("--sample-id", type=int, default=0, help="Sample index in dataset")
    parser.add_argument("--max-vel", type=float, default=1.0, help="Robot max velocity for AVOCADO")
    parser.add_argument("--agent-radius", type=float, default=0.2, help="Robot radius")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--c", type=float, default=0.7)
    parser.add_argument("--d", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=14.15)
    parser.add_argument("--epsilon", type=float, default=3.22)
    parser.add_argument("--delta", type=float, default=0.57)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--animate", action="store_true", help="Animate trajectories")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    human_positions_all = data["human_positions"]
    human_velocities_all = data["human_velocities"]
    robot_starts_all = data["robot_starts"]
    robot_goals_all = data["robot_goals"]
    scenarios = data["scenarios"]
    speed_scales = data["speed_scales"]
    dt = float(data["dt"])

    n_samples = human_positions_all.shape[0]
    if args.sample_id < 0 or args.sample_id >= n_samples:
        raise IndexError(f"sample-id out of range [0, {n_samples - 1}]")

    human_positions = human_positions_all[args.sample_id]
    human_velocities = human_velocities_all[args.sample_id]
    robot_start = robot_starts_all[args.sample_id]
    robot_goal = robot_goals_all[args.sample_id]

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

    robot_avocado_positions, robot_avocado_velocities = _run_avocado_rollout(
        human_positions=human_positions,
        human_velocities=human_velocities,
        robot_start=robot_start,
        robot_goal=robot_goal,
        dt=dt,
        max_vel=args.max_vel,
        actor_params=actor_params,
    )

    min_dist_avocado = _compute_min_dist(robot_avocado_positions, human_positions)

    scenario = scenarios[args.sample_id]
    speed = speed_scales[args.sample_id]
    title = f"sample={args.sample_id} | scenario={scenario} | speed={speed:.2f}"

    print("Replay summary:")
    print(f"  sample_id: {args.sample_id}")
    print(f"  scenario: {scenario}")
    print(f"  speed_scale: {speed:.2f}")
    print(f"  min_dist_avocado: {min_dist_avocado:.4f}")
    print(f"  robot_start: {robot_start}")
    print(f"  robot_goal: {robot_goal}")
    print(f"  final_robot_avocado: {robot_avocado_positions[-1]}")
    print(f"  mean_speed_avocado: {np.linalg.norm(robot_avocado_velocities, axis=1).mean():.4f}")

    if args.animate:
        _animate(
            human_positions=human_positions,
            robot_avocado_positions=robot_avocado_positions,
            robot_start=robot_start,
            robot_goal=robot_goal,
            title=title,
            dt=dt,
        )
    else:
        _plot_static(
            human_positions=human_positions,
            robot_avocado_positions=robot_avocado_positions,
            robot_start=robot_start,
            robot_goal=robot_goal,
            title=title,
        )


if __name__ == "__main__":
    main()
