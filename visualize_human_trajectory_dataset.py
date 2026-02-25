import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def _line_trajectory(start, goal, num_steps):
    xs = np.linspace(start[0], goal[0], num_steps)
    ys = np.linspace(start[1], goal[1], num_steps)
    return np.stack((xs, ys), axis=1)


def _plot_static(ax, human_pos, robot_pos, title):
    ax.plot(human_pos[:, 0], human_pos[:, 1], "r-", lw=2, label="Human")
    ax.plot(robot_pos[:, 0], robot_pos[:, 1], "b-", lw=2, label="Robot")
    ax.scatter(human_pos[0, 0], human_pos[0, 1], c="r", marker="o", s=45)
    ax.scatter(human_pos[-1, 0], human_pos[-1, 1], c="r", marker="x", s=65)
    ax.scatter(robot_pos[0, 0], robot_pos[0, 1], c="b", marker="o", s=45)
    ax.scatter(robot_pos[-1, 0], robot_pos[-1, 1], c="b", marker="x", s=65)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def _animate_sample(human_pos, robot_pos, title, dt):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    min_x = min(human_pos[:, 0].min(), robot_pos[:, 0].min()) - 0.5
    max_x = max(human_pos[:, 0].max(), robot_pos[:, 0].max()) + 0.5
    min_y = min(human_pos[:, 1].min(), robot_pos[:, 1].min()) - 0.5
    max_y = max(human_pos[:, 1].max(), robot_pos[:, 1].max()) + 0.5
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    human_path, = ax.plot([], [], "r-", lw=2, label="Human")
    robot_path, = ax.plot([], [], "b-", lw=2, label="Robot")
    human_dot, = ax.plot([], [], "ro", ms=6)
    robot_dot, = ax.plot([], [], "bo", ms=6)
    ax.legend(loc="upper right")

    def init():
        human_path.set_data([], [])
        robot_path.set_data([], [])
        human_dot.set_data([], [])
        robot_dot.set_data([], [])
        return human_path, robot_path, human_dot, robot_dot

    def update(frame):
        human_path.set_data(human_pos[: frame + 1, 0], human_pos[: frame + 1, 1])
        robot_path.set_data(robot_pos[: frame + 1, 0], robot_pos[: frame + 1, 1])
        human_dot.set_data(human_pos[frame, 0], human_pos[frame, 1])
        robot_dot.set_data(robot_pos[frame, 0], robot_pos[frame, 1])
        return human_path, robot_path, human_dot, robot_dot

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=human_pos.shape[0],
        interval=dt * 1000,
        blit=True,
        repeat=False,
    )
    plt.show()
    return anim


def _pick_indices_per_scenario(scenarios, per_scenario, rng):
    unique_scenarios = sorted(set(str(s) for s in scenarios))
    selected = []
    for scenario in unique_scenarios:
        idx = np.where(np.array([str(s) for s in scenarios]) == scenario)[0]
        if len(idx) == 0:
            continue
        n_pick = min(per_scenario, len(idx))
        selected.extend(rng.choice(idx, size=n_pick, replace=False).tolist())
    return np.array(selected, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Visualize random samples from human/robot trajectory dataset.")
    parser.add_argument("--dataset", default="data/human_robot_trajectories.npz", help="Path to dataset .npz")
    parser.add_argument("--n-samples", type=int, default=4, help="How many random samples to plot")
    parser.add_argument("--per-scenario", type=int, default=None, help="Samples per scenario (overrides --n-samples)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    parser.add_argument("--animate", action="store_true", help="Animate selected samples one by one")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    human_positions = data["human_positions"]
    if "robot_positions" in data:
        robot_positions = data["robot_positions"]
    elif "robot_starts" in data and "robot_goals" in data:
        robot_starts = data["robot_starts"]
        robot_goals = data["robot_goals"]
        robot_positions = np.stack(
            [_line_trajectory(robot_starts[i], robot_goals[i], human_positions.shape[1]) for i in range(human_positions.shape[0])],
            axis=0,
        )
    else:
        raise KeyError("Dataset must contain either robot_positions or (robot_starts and robot_goals).")
    scenarios = data["scenarios"]
    speed_scales = data["speed_scales"]
    dt = float(data["dt"])

    n_total = human_positions.shape[0]
    if n_total == 0:
        raise RuntimeError("Dataset is empty.")

    rng = np.random.default_rng(args.seed)
    if args.per_scenario is not None:
        indices = _pick_indices_per_scenario(scenarios, args.per_scenario, rng)
        n_pick = len(indices)
    else:
        n_pick = min(args.n_samples, n_total)
        indices = rng.choice(n_total, size=n_pick, replace=False)

    if args.animate:
        for sample_id in indices:
            scenario = scenarios[sample_id]
            speed = speed_scales[sample_id]
            title = f"sample={sample_id} | scenario={scenario} | speed={speed:.2f}"
            _animate_sample(human_positions[sample_id], robot_positions[sample_id], title, dt)
    else:
        n_cols = 2 if n_pick > 1 else 1
        n_rows = int(np.ceil(n_pick / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        axes = np.array(axes).reshape(-1)

        for k, sample_id in enumerate(indices):
            scenario = scenarios[sample_id]
            speed = speed_scales[sample_id]
            title = f"sample={sample_id} | scenario={scenario} | speed={speed:.2f}"
            _plot_static(axes[k], human_positions[sample_id], robot_positions[sample_id], title)

        for k in range(n_pick, len(axes)):
            axes[k].axis("off")

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
