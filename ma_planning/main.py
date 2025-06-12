import pygame
import numpy as np
from pygame_env_utils import Robot, draw_obstacles, update_moving_obstacles
from obstacle_generator import generate_mixed_scene
from controller import MPPIController, GlobalController
from dynamic_models import DifferentialDriveDynamics
from motion_planner import *
MAP_WIDTH_M = 50.0
MAP_HEIGHT_M = 50.0
SCREEN_WIDTH_PX = 1000
SCREEN_HEIGHT_PX = 1000
SCALE = SCREEN_WIDTH_PX / MAP_WIDTH_M
FPS = 60

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX))
clock = pygame.time.Clock()

obstacles = generate_mixed_scene(MAP_WIDTH_M, MAP_HEIGHT_M)
diffdrive = DifferentialDriveDynamics()

# Parameters
num_robots = 5
circle_radius = 13
circle_center = np.array([MAP_WIDTH_M / 2, MAP_HEIGHT_M / 2])

circle_center = (25, 25)
circle_radius = 20
num_robots = 5
diffdrive = DifferentialDriveDynamics()



robots = create_robots_with_goals(
    num_robots=num_robots,
    center=circle_center,
    radius=circle_radius,
    dynamics_model=diffdrive,
)

map_bounds = (0, 0, MAP_WIDTH_M, MAP_HEIGHT_M)

mamppi_controller = MAMPPIController(model=diffdrive, horizon=100, 
                                    stepsize=0.02,
                                    num_samples=400,
                                    lambda_=0.01,
                                    noise_std=np.array([0.1,0.02]),
                                    num_robots=num_robots)

use_mamppi = False  # ← SWITCH THIS TO COMPARE CONTROLLERS

def step():
    dt = clock.tick(FPS) / 1000.0


    # --- check if MA or SA mode ---
    if use_mamppi:
        states = np.array([r.state for r in robots])
        goals = np.array([r.goal for r in robots])
        occupancy_grids = [None for _ in robots]  # or use real ones if needed

        joint_actions = mamppi_controller.compute_actions(states, goals, occupancy_grids)
        
        for i, robot in enumerate(robots):
            robot.update_state(joint_actions[i], dt)
            robot.set_planned_trajectory(mamppi_controller.optimal_rollouts[i])

    else:
        for robot in robots:
            robot.update(dt, obstacles)
    # --- mode check end ---


    update_moving_obstacles(obstacles, dt, map_bounds)

    screen.fill((255, 255, 255))
    draw_obstacles(screen, obstacles, SCALE, SCREEN_HEIGHT_PX)
    for robot in robots:
        robot.draw(screen, SCALE, SCREEN_HEIGHT_PX)

    pygame.display.flip()
    return dt

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        step()
    pygame.quit()

if __name__ == "__main__":
    main()