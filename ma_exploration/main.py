# main.py
import pygame
from pygame_env_utils import Robot, draw_obstacles, update_moving_obstacles
from obstacle_generator import generate_mixed_scene

from controller import *

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


"""
controller = MPPIController(
    model=None,  # use a simple forward kinematic model
    horizon=15,
    num_samples=100,
    lambda_=1.0,
)
"""

controller1 = GlobalController()
robot1 = Robot(5, 5, 0, goal=(45, 45), sensor_range=10.0, controller=controller1)
controller2 = GlobalController()
robot2 = Robot(5, 45, 0, goal=(45, 45), sensor_range=10.0, controller=controller1)

map_bounds = (0, 0, MAP_WIDTH_M, MAP_HEIGHT_M)

def step():
    dt = clock.tick(FPS) / 1000.0
    robot1.update(dt, obstacles)
    robot2.update(dt, obstacles)

    update_moving_obstacles(obstacles, dt, map_bounds)

    screen.fill((255, 255, 255))
    draw_obstacles(screen, obstacles, SCALE, SCREEN_HEIGHT_PX)
    robot1.draw(screen, SCALE, SCREEN_HEIGHT_PX)
    robot2.draw(screen, SCALE, SCREEN_HEIGHT_PX)
    pygame.display.flip()
    return dt


def reset():
    global robot1, robot2, obstacles
    robot1 = Robot(5, 5, 0, goal=(45, 45), sensor_range=5.0)
    robot2 = Robot(5, 5, 0, goal=(45, 45), sensor_range=5.0)
    obstacles = generate_mixed_scene(MAP_WIDTH_M, MAP_HEIGHT_M)


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