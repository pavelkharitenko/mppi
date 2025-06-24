import pygame
import math
import numpy as np
from planner import Planner
from controller import *
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (180, 255, 180)
LIGHT_BLUE = (150, 200, 255)
GRAY = (220, 220, 220)


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
    for x, y, r, vx, vy in obstacles["moving"]:
        x_px, y_px = world_to_screen(x, y, screen_height_px, scale)
        pygame.draw.circle(screen, RED, (x_px, y_px), int(r * scale))


def update_moving_obstacles(obstacles, dt, map_bounds):
    x_min, y_min, x_max, y_max = map_bounds
    for i, (x,y,r,vx,vy) in enumerate(obstacles["moving"]):

        x += vx * dt
        y += vy * dt

        if x - r < x_min or x + r > x_max:
            vx *= -1
        if y - r < y_min or y + r > y_max:
            vy *= -1

        
        obstacles["moving"][i] = (x,y, r,vx, vy)

class Robot:
    def __init__(self, x, y, theta, goal, sensor_range, dynamics_model, controller=None, radius=0.5):
        self.state = np.array([x, y, theta])
        self.radius = radius
        self.target = goal
        self.sensor_range = sensor_range
        self.model = dynamics_model
        self.controller = controller
        self.planned_trajectory = np.array([])
        self.planner = Planner(goal, sensor_range)
        

    def update(self, dt, obstacles):
        """
        If robot has own controller.
        """
        
        self.planner.update_local_map(self.state, obstacles)
        local_goal = self.planner.plan((self.state[0], self.state[1]))
        grid_info = self.planner.grid_info


        u = self.controller.compute_action(self.state, local_goal, grid_info)

        # Save planned trajectory if controller supports it
        if hasattr(self.controller, "optimal_rollout"):
            self.planned_trajectory = self.controller.optimal_rollout[0]

        self.state = self.model.step(self.state, u, dt)

    def update_state(self, control, dt):
        """
        If high-level multi-agent controller computed control input, just update state.
        """
        self.state = self.model.step(self.state, control, dt)


    def set_planned_trajectory(self, trajectory):
        self.planned_trajectory = trajectory


    def draw(self, screen, scale, screen_height_px):
        x, y, theta = self.state
        x_px, y_px = world_to_screen(x, y, screen_height_px, scale)

        # Draw sensor range if enabled
        pygame.draw.circle(screen, LIGHT_BLUE, (x_px, y_px), int(self.sensor_range * scale), width=1)

        # Draw global goal
        gx, gy = self.target
        gx_px, gy_px = world_to_screen(gx, gy, screen_height_px, scale)
        pygame.draw.circle(screen, LIGHT_GREEN, (gx_px, gy_px), 6)

        # Draw local goal 
        tx, ty = self.planner.plan((x, y))
        tx_px, ty_px = world_to_screen(tx, ty, screen_height_px, scale)
        pygame.draw.circle(screen, GREEN, (tx_px, ty_px), 4)

        # Draw robot
        pygame.draw.circle(screen, GREEN, (x_px, y_px), int(self.radius * scale))
        dx = math.cos(theta) * self.radius * scale * 2
        dy = -math.sin(theta) * self.radius * scale * 2
        pygame.draw.line(screen, BLACK, (x_px, y_px), (x_px + int(dx), y_px + int(dy)), 2)

        # Draw occupancy grid if local planning
        if self.planner.grid:
            size = len(self.planner.grid)
            cell_size = self.planner.map_resolution * scale
            for i in range(size):
                for j in range(size):
                    if self.planner.grid[i][j]:
                        wx = x + (j * self.planner.map_resolution - self.sensor_range)
                        #wx = x - self.sensor_range + j * self.planner.grid_res + self.planner.grid_res/2
                        wy = y + (i * self.planner.map_resolution - self.sensor_range)
                        #wy = y - self.sensor_range + i * self.planner.grid_res + self.planner.grid_res/2

                        rect_x, rect_y = world_to_screen(wx, wy, screen_height_px, scale)
                        rect = pygame.Rect(rect_x, rect_y, cell_size, cell_size)
                        pygame.draw.rect(screen, GRAY, rect)

        # Draw planned trajectory (optional)
        if self.planned_trajectory is not None and len(self.planned_trajectory) > 0:
            #print("planned traj", self.planned_trajectory)
            for i in range(len(self.planned_trajectory) - 1):
                x1, y1, _ = self.planned_trajectory[i]
                x2, y2, _ = self.planned_trajectory[i + 1]
                p1 = world_to_screen(x1, y1, screen_height_px, scale)
                p2 = world_to_screen(x2, y2, screen_height_px, scale)
                pygame.draw.line(screen, (0, 0, 255), p1, p2, 3)  # blue trajectory

