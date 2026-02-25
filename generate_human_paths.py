import numpy as np


def generate_human_paths(n_per_scenario, scenarios, n_waypoints=8, workspace=None, rng=None):
    if workspace is None:
        workspace = np.array([[-4.5, 4.5], [-4.5, 4.5]], dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    human_data = []

    for scenario in scenarios:
        for _ in range(n_per_scenario):
            if scenario == "crossing":
                y0 = -1.5 + 3 * rng.random()
                start = np.array([workspace[0, 0], y0 + 0.2 * rng.normal()])
                goal = np.array([workspace[0, 1], y0 + 0.2 * rng.normal()])
                xs = np.linspace(start[0], goal[0], n_waypoints)
                ys = np.linspace(start[1], goal[1], n_waypoints)
                if rng.random() < 0.5:
                    curvature = 0.3 + 0.5 * rng.random()
                    if rng.random() < 0.5:
                        curvature = -curvature
                    ys = ys + curvature * np.sin(np.linspace(0, np.pi, len(xs)))

            elif scenario == "meeting":
                y0 = -1.5 + 3 * rng.random()
                start = np.array([workspace[0, 0], y0 + 0.1 * rng.normal()])
                goal = np.array([workspace[0, 1], y0 + 0.1 * rng.normal()])
                xs = np.linspace(start[0], goal[0], n_waypoints)
                ys = np.linspace(start[1], goal[1], n_waypoints)
                if rng.random() < 0.5:
                    curvature = 0.5 + 0.4 * rng.random()
                    if rng.random() < 0.5:
                        curvature = -curvature
                    ys = ys + curvature * np.sin(np.linspace(0, np.pi, len(xs)))

            elif scenario == "meeting_delayed_arc":
                y0 = -1.5 + 3 * rng.random()
                start = np.array([workspace[0, 0], y0 + 0.1 * rng.normal()])
                goal = np.array([workspace[0, 1], y0 + 0.1 * rng.normal()])
                xs = np.linspace(start[0], goal[0], n_waypoints)
                ys = np.linspace(start[1], goal[1], n_waypoints)
                turn_start_idx = rng.integers(int(np.ceil(n_waypoints / 3.0)), n_waypoints - 1)
                arc_magnitude = 1.0 + 0.5 * rng.random()
                arc_xs = xs[turn_start_idx:]
                arc_ys = arc_magnitude * np.sin(np.linspace(0, np.pi, len(arc_xs)))
                ys[turn_start_idx:] = ys[turn_start_idx:] + arc_ys

            elif scenario == "abrupt_change":
                start_above = rng.random() < 0.5
                if start_above:
                    y_start = 1.5 + 1.5 * rng.random()
                    y_end = -1.5 - 1.5 * rng.random()
                else:
                    y_start = -1.5 - 1.5 * rng.random()
                    y_end = 1.5 + 1.5 * rng.random()

                start = np.array([workspace[0, 0], y_start + 0.2 * rng.normal()])
                mid_x = -1.5 + 3 * rng.random()
                mid = np.array([mid_x, y_start + 0.1 * rng.normal()])
                goal = np.array([workspace[0, 1], y_end + 0.2 * rng.normal()])

                xs = np.concatenate(
                    [
                        np.linspace(start[0], mid[0], int(np.ceil(n_waypoints / 2.0))),
                        np.linspace(mid[0], goal[0], int(np.floor(n_waypoints / 2.0)) + 1),
                    ]
                )
                ys = np.concatenate(
                    [
                        np.linspace(start[1], mid[1], int(np.ceil(n_waypoints / 2.0))),
                        np.linspace(mid[1], goal[1], int(np.floor(n_waypoints / 2.0)) + 1),
                    ]
                )
                xs = -xs

            elif scenario == "step_pattern":
                y0 = -1.5 + 3 * rng.random()
                start = np.array([workspace[0, 0], y0 + 0.2 * rng.normal()])
                goal = np.array([workspace[0, 1], y0 + 0.2 * rng.normal()])
                xs = np.linspace(start[0], goal[0], n_waypoints)
                ys = np.linspace(start[1], goal[1], n_waypoints)
                step_size = 0.2 + rng.random()
                step_pos = rng.integers(1, n_waypoints - 1)
                ys[step_pos:] = ys[step_pos:] + step_size * np.sign(rng.normal())
                ys = ys + 0.3 * rng.normal(size=ys.shape)
                xs = xs + 0.3 * rng.normal(size=xs.shape)
                xs = -xs

            elif scenario == "zigzag":
                y0 = -1.5 + 3 * rng.random()
                start = np.array([workspace[0, 0], y0 + 0.2 * rng.normal()])
                goal = np.array([workspace[0, 1], y0 + 0.2 * rng.normal()])
                xs = np.linspace(start[0], goal[0], n_waypoints)
                ys = start[1] + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n_waypoints)) + 0.2 * rng.normal(size=xs.shape)
                xs = xs + 0.2 * rng.normal(size=xs.shape)
                xs = -xs

            else:
                raise ValueError("Unknown scenario.")

            min_len = min(len(xs), len(ys))
            path = np.vstack((xs[:min_len], ys[:min_len]))
            human_data.append({"scenario": scenario, "path": path})

    return human_data
