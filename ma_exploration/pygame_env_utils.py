import pygame
import math
from planner import Planner

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (180, 255, 180)
LIGHT_BLUE = (150, 200, 255)


def world_to_screen(x_m, y_m, screen_height_px, scale):
    x_px = int(x_m * scale)
    y_px = int(screen_height_px - y_m * scale)
    return x_px, y_px


def screen_to_world(x_px, y_px, screen_height_px, scale):
    x_m = x_px / scale
    y_m = (screen_height_px - y_px) / scale
    return x_m, y_m


def draw_obstacles(screen, obstacles, scale, screen_height_px):
    for x, y, r in obstacles["static"]:
        x_px, y_px = world_to_screen(x, y, screen_height_px, scale)
        pygame.draw.circle(screen, BLACK, (x_px, y_px), int(r * scale))
    for mob in obstacles["moving"]:
        x_px, y_px = world_to_screen(*mob["pos"], screen_height_px, scale)
        pygame.draw.circle(screen, RED, (x_px, y_px), int(mob["radius"] * scale))


def update_moving_obstacles(obstacles, dt, map_bounds):
    x_min, y_min, x_max, y_max = map_bounds
    for mob in obstacles["moving"]:
        x, y = mob["pos"]
        vx, vy = mob["vel"]

        x += vx * dt
        y += vy * dt

        if x - mob["radius"] < x_min or x + mob["radius"] > x_max:
            vx *= -1
        if y - mob["radius"] < y_min or y + mob["radius"] > y_max:
            vy *= -1

        mob["pos"] = [x, y]
        mob["vel"] = [vx, vy]


class Robot:
    def __init__(self, x, y, theta, goal, sensor_range, radius=0.5):
        self.x, self.y, self.theta = x, y, theta
        self.v = 0
        self.w = 0
        self.radius = radius
        self.planner = Planner(goal, sensor_range)
        self.target = goal
        self.sensor_range = sensor_range

    def update(self, dt):
        tx, ty = self.planner.plan((self.x, self.y))
        angle_to_target = math.atan2(ty - self.y, tx - self.x)
        angle_diff = angle_to_target - self.theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        self.v = 1.0
        self.w = 2.0 * angle_diff

        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.w * dt

    def draw(self, screen, scale, screen_height_px):
        x_px, y_px = world_to_screen(self.x, self.y, screen_height_px, scale)

        # Draw sensor range circle
        pygame.draw.circle(screen, LIGHT_BLUE, (x_px, y_px), int(self.sensor_range * scale), width=1)

        # Draw global goal
        gx, gy = self.target
        gx_px, gy_px = world_to_screen(gx, gy, screen_height_px, scale)
        pygame.draw.circle(screen, LIGHT_GREEN, (gx_px, gy_px), 6)

        # Draw local goal
        tx, ty = self.planner.plan((self.x, self.y))
        tx_px, ty_px = world_to_screen(tx, ty, screen_height_px, scale)
        pygame.draw.circle(screen, GREEN, (tx_px, ty_px), 4)

        # Draw robot
        pygame.draw.circle(screen, GREEN, (x_px, y_px), int(self.radius * scale))
        dx = math.cos(self.theta) * self.radius * scale * 2
        dy = -math.sin(self.theta) * self.radius * scale * 2
        pygame.draw.line(screen, BLACK, (x_px, y_px), (x_px + int(dx), y_px + int(dy)), 2)