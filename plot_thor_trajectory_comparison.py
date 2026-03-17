import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from evaluate_dataset_avocado import _run_avocado_rollout
from main_opinion_ttc_avocado_dataset import get_opinion_params, _run_opinion_rollout_from_dataset


def _parse_sample_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _pick_samples(n_total: int, explicit_ids: List[int], n_random: int, seed: int) -> List[int]:
    valid_explicit = [idx for idx in explicit_ids if 0 <= idx < n_total]
    if valid_explicit:
        return sorted(set(valid_explicit))

    n_random = max(1, min(int(n_random), n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=n_random, replace=False).tolist())


def _plot_one_sample(
    sample_id: int,
    scenario: str,
    human_positions: np.ndarray,
    robot_start: np.ndarray,
    robot_goal: np.ndarray,
    traj_kappa: np.ndarray,
    traj_ttc: np.ndarray,
    traj_avocado: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    if human_positions.ndim == 2:
        ax.plot(human_positions[:, 0], human_positions[:, 1], color="tab:blue", linewidth=2.0, label="human")
        ax.scatter(human_positions[0, 0], human_positions[0, 1], color="tab:blue", marker="o", s=30)
        ax.scatter(human_positions[-1, 0], human_positions[-1, 1], color="tab:blue", marker="x", s=36)
    else:
        for idx in range(human_positions.shape[1]):
            label = "human" if idx == 0 else None
            traj = human_positions[:, idx, :]
            ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.0, alpha=0.65, label=label)

    ax.plot(traj_kappa[:, 0], traj_kappa[:, 1], color="tab:red", linewidth=2.0, label="baseline")
    ax.plot(traj_ttc[:, 0], traj_ttc[:, 1], color="tab:green", linewidth=2.0, label="baseline+ttc")
    ax.plot(traj_avocado[:, 0], traj_avocado[:, 1], color="tab:orange", linewidth=2.0, label="avocado")

    ax.scatter(robot_start[0], robot_start[1], color="black", marker="^", s=55, label="robot start")
    ax.scatter(robot_goal[0], robot_goal[1], color="black", marker="*", s=90, label="robot goal")

    ax.set_title(f"sample {sample_id} | scenario: {scenario}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trajectory overlays for baseline, baseline+ttc, and avocado.")
    parser.add_argument("--dataset", default="data/dataset_thor_opinion_converted.npz")
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids, e.g. 0,10,20")
    parser.add_argument("--num-random", type=int, default=6, help="Used when --sample-ids is empty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="data/trajectory_plots_thor_comparison")
    parser.add_argument("--goal-tolerance", type=float, default=0.2)
    parser.add_argument("--collision-distance", type=float, default=0.4)
    parser.add_argument("--extra-steps", type=int, default=40)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--dr", type=float, default=2.4)
    parser.add_argument("--alpha-r", type=float, default=0.3)
    parser.add_argument("--gamma-r", type=float, default=10.0)
    parser.add_argument("--b-r", type=float, default=0.0)
    parser.add_argument("--Rr", type=float, default=6.0)
    parser.add_argument("--kr", type=float, default=1.5)
    parser.add_argument("--beta-r", type=float, default=float(np.pi / 4.0))
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--tau-u", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--c", type=float, default=0.7)
    parser.add_argument("--d", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=14.15)
    parser.add_argument("--epsilon", type=float, default=3.22)
    parser.add_argument("--delta", type=float, default=0.57)
    parser.add_argument("--bias", type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    human_positions_all = data["human_positions"]
    human_velocities_all = data["human_velocities"]
    robot_starts_all = data["robot_starts"]
    robot_goals_all = data["robot_goals"]
    scenarios = data["scenarios"]
    robot_speeds = data["robot_speeds"] if "robot_speeds" in data.files else None
    dt = float(data["dt"])

    n_samples = int(human_positions_all.shape[0])
    chosen = _pick_samples(n_samples, _parse_sample_ids(args.sample_ids), args.num_random, args.seed)

    opinion_params = get_opinion_params(
        {
            "dr": args.dr,
            "alpha_r": args.alpha_r,
            "gamma_r": args.gamma_r,
            "b_r": args.b_r,
            "Rr": args.Rr,
            "kr": args.kr,
            "beta_r": args.beta_r,
            "u_max": args.u_max,
            "u_min": args.u_min,
            "n": args.n,
            "tau_u": args.tau_u,
        }
    )
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

    print(f"Plotting {len(chosen)} samples into: {output_dir}")
    for sample_id in chosen:
        human_positions = human_positions_all[sample_id]
        human_velocities = human_velocities_all[sample_id]
        robot_start = np.asarray(robot_starts_all[sample_id], dtype=float).reshape(2)
        robot_goal = np.asarray(robot_goals_all[sample_id], dtype=float).reshape(2)
        scenario = str(scenarios[sample_id])
        robot_speed = float(robot_speeds[sample_id]) if robot_speeds is not None else float(args.max_vel)

        traj_kappa = _run_opinion_rollout_from_dataset(
            human_positions=human_positions,
            human_velocities=human_velocities,
            robot_start=robot_start,
            robot_goal=robot_goal,
            dt=dt,
            robot_speed=robot_speed,
            opinion_params=opinion_params,
            attention_mode="kappa",
            collision_distance=args.collision_distance,
            goal_tolerance=args.goal_tolerance,
            extra_steps=args.extra_steps,
        )
        traj_ttc = _run_opinion_rollout_from_dataset(
            human_positions=human_positions,
            human_velocities=human_velocities,
            robot_start=robot_start,
            robot_goal=robot_goal,
            dt=dt,
            robot_speed=robot_speed,
            opinion_params=opinion_params,
            attention_mode="ttc",
            collision_distance=args.collision_distance,
            goal_tolerance=args.goal_tolerance,
            extra_steps=args.extra_steps,
        )
        traj_avocado = _run_avocado_rollout(
            human_positions=human_positions,
            human_velocities=human_velocities,
            robot_start=robot_start,
            robot_goal=robot_goal,
            dt=dt,
            max_vel=robot_speed,
            actor_params=actor_params,
            extra_steps=args.extra_steps,
        )

        out_file = output_dir / f"sample_{sample_id:04d}_{scenario}.png"
        _plot_one_sample(
            sample_id=sample_id,
            scenario=scenario,
            human_positions=human_positions,
            robot_start=robot_start,
            robot_goal=robot_goal,
            traj_kappa=traj_kappa,
            traj_ttc=traj_ttc,
            traj_avocado=traj_avocado,
            output_path=out_file,
        )
        print(f"  wrote: {out_file}")


if __name__ == "__main__":
    main()
