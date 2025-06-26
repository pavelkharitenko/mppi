from controller import *
from pygame_env_utils import *

def generate_circle_positions(num_robots, center, radius):
    """Returns a list of (x, y, theta) tuples spaced evenly on a circle."""
    positions = []
    for i in range(num_robots):
        angle = 2 * np.pi * i / num_robots
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        theta = angle + np.pi  # Facing inward
        positions.append((x, y, theta))
    return positions

def assign_antipodal_goals(positions):
    """Returns a list of (goal_x, goal_y) pairs based on antipodal assignment."""
    num = len(positions)
    goals = []
    for i in range(num):
        antipodal_idx = (i + num // 2) % num
        gx, gy, _ = positions[antipodal_idx]
        goals.append((gx, gy))
    return goals

def create_circle_robots_with_antipolar_goals(num_robots, center, radius, dynamics_model, use_single_agent_controller=True):
    """
    Creates and returns a list of Robot instances with antipodal goals.
    
    """
    start_positions = generate_circle_positions(num_robots, center, radius)
    goal_positions = assign_antipodal_goals(start_positions)
    start_and_goals = list(zip(start_positions, goal_positions))

    robots = []
    for i in range(num_robots):
        if use_single_agent_controller:
            controller = MPPIController(
                model=dynamics_model,
                horizon=150,
                stepsize=1.0/60.0,
                num_samples=550,
                lambda_=35.0,
                noise_std=np.array([0.4, 1.2])
            )
        else:
            controller = None  # MAMPPI will be used instead

        (x, y, theta), (gx, gy) = start_and_goals[i]
        robot = Robot(
            x, y, theta,
            goal=(gx, gy),
            sensor_range=10.0,
            dynamics_model=dynamics_model,
            controller=controller,
        )
        robots.append(robot)


    return robots

def generate_circle_positions_tangential(num_robots, center, radius, clockwise=True):
    """
    Returns a list of (x, y, theta) tuples spaced evenly on a circle.
    theta is tangential:   CW  →  θ = angle - π/2
                           CCW →  θ = angle + π/2
    """
    poses = []
    for i in range(num_robots):
        angle = 2.0 * np.pi * i / num_robots           # radial direction
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        if clockwise:
            theta = angle - np.pi / 2.0                # tangent CW
        else:
            theta = angle + np.pi / 2.0                # tangent CCW

        poses.append((x, y, theta))
    return poses



def create_circle_robots_tangential(num_robots,
                             center,
                             radius,
                             dynamics_model,
                             clockwise=True,
                             use_single_agent_controller=True):
    """
    Creates robots on a circle, tangential heading, antipodal goals.
    """
    start_positions = generate_circle_positions_tangential(
        num_robots, center, radius, clockwise=clockwise
    )
    goal_positions = assign_antipodal_goals(start_positions)

    robots = []
    for i, ((x, y, theta), (gx, gy)) in enumerate(zip(start_positions, goal_positions)):

        if use_single_agent_controller:
            controller = MPPIController(
                model=dynamics_model,
                horizon=200,
                stepsize=1.0/60.0,
                num_samples=700,
                lambda_=45.0,
                noise_std=np.array([0.4, 1.2])
            )
        else:
            controller = None  # shared MA controller

        robots.append(
            Robot(
                x, y, theta,
                goal=(gx, gy),
                sensor_range=6.0,
                dynamics_model=dynamics_model,
                controller=controller,
            )
        )

    return robots



def create_circle_robots_tangential_sa_d4orm(num_robots,
                             center,
                             radius,
                             dynamics_model,
                             clockwise=True,
                             use_single_agent_controller=True):
    """
    Creates robots on a circle, tangential heading, antipodal goals.
    """
    start_positions = generate_circle_positions_tangential(
        num_robots, center, radius, clockwise=clockwise
    )
    goal_positions = assign_antipodal_goals(start_positions)

    robots = []
    for i, ((x, y, theta), (gx, gy)) in enumerate(zip(start_positions, goal_positions)):

        if use_single_agent_controller:
            controller = D4ormSAController(
                model=dynamics_model,
                horizon=200,
                diffusion_steps=20,
                stepsize=1.0/60.0,
                num_samples=3,
                lambda_=45.0,
                alpha_decay=0.96
            )
        else:
            controller = None  # shared MA controller

        robots.append(
            Robot(
                x, y, theta,
                goal=(gx, gy),
                sensor_range=6.0,
                dynamics_model=dynamics_model,
                controller=controller,
            )
        )

    return robots
