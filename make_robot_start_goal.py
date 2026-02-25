import numpy as np


def make_robot_start_goal(scenario, human_path, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if scenario == "crossing":
        mid_x = np.mean(human_path[0, :])
        robot_start = np.array([mid_x, -4.5])
        robot_goal = np.array([mid_x, 4.5])

    elif scenario == "meeting":
        human_start_y = human_path[1, 0]
        robot_start = np.array([4.5, human_start_y])
        robot_goal = np.array([-4.5, human_start_y])
        robot_start[1] = robot_start[1] + 0.1 * rng.normal()
        robot_goal[1] = robot_goal[1] + 0.1 * rng.normal()

    elif scenario == "meeting_delayed_arc":
        human_start_y = human_path[1, 0]
        robot_start = np.array([4.5, human_start_y])
        robot_goal = np.array([-4.5, human_start_y])
        robot_start[1] = robot_start[1] + 0.1 * rng.normal()
        robot_goal[1] = robot_goal[1] + 0.1 * rng.normal()

    elif scenario == "abrupt_change":
        human_start_y = human_path[1, 0]
        robot_start = np.array([-4.5, human_start_y])
        robot_goal = np.array([4.5, human_start_y])

    elif scenario == "step_pattern":
        mid_y = np.mean(human_path[1, :])
        robot_start = np.array([-4.5, mid_y])
        robot_goal = np.array([4.5, mid_y])

    elif scenario == "zigzag":
        mid_y = np.mean(human_path[1, :])
        robot_start = np.array([-4.5, mid_y])
        robot_goal = np.array([4.5, mid_y])

    else:
        raise ValueError("Unknown scenario.")

    return robot_start, robot_goal
