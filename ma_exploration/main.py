# main.py
import pygame
from pygame_env_utils import Robot, draw_obstacles, update_moving_obstacles
from obstacle_generator import generate_mixed_scene

MAP_WIDTH_M = 100.0
MAP_HEIGHT_M = 100.0
SCREEN_WIDTH_PX = 1000
SCREEN_HEIGHT_PX = 1000
SCALE = SCREEN_WIDTH_PX / MAP_WIDTH_M
FPS = 60

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX))
clock = pygame.time.Clock()

obstacles = generate_mixed_scene(MAP_WIDTH_M, MAP_HEIGHT_M)
robot = Robot(5, 5, 0, goal=(45, 45), sensor_range=10.0)
map_bounds = (0, 0, MAP_WIDTH_M, MAP_HEIGHT_M)

def step():
    dt = clock.tick(FPS) / 1000.0
    robot.update(dt)
    update_moving_obstacles(obstacles, dt, map_bounds)

    screen.fill((255, 255, 255))
    draw_obstacles(screen, obstacles, SCALE, SCREEN_HEIGHT_PX)
    robot.draw(screen, SCALE, SCREEN_HEIGHT_PX)
    pygame.display.flip()
    return dt


def reset():
    global robot, obstacles
    robot = Robot(5, 5, 0, goal=(45, 45), sensor_range=10.0)
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