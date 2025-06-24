import math
from itertools import chain
import numpy as np

from dataclasses import dataclass

@dataclass
class GridInfo:
    occ: np.ndarray
    half: float
    res: float
    S: int


class Planner:
    def __init__(self, goal, sensor_range, map_resolution=0.25):
        self.goal = goal  # (x_goal, y_goal)
        self.sensor_range = sensor_range  # in meters
        self.map_resolution = map_resolution
        #self.grid_res = 0.25 * self.map_resolution
        self.grid = None
        self.grid_np = None
        self.grid_info = None


    def plan(self, robot_pos):
        x, y = robot_pos
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        distance = math.hypot(dx, dy)

        # If goal is within sensor range, go directly
        if distance <= self.sensor_range:
            return gx, gy

        # Else, go towards goal but only up to sensor range
        angle = math.atan2(dy, dx)
        next_x = x + self.sensor_range * math.cos(angle)
        next_y = y + self.sensor_range * math.sin(angle)
        return next_x, next_y

    def update_local_map(self, robot_pos, obstacles):
        x, y, theta = robot_pos
        half = self.sensor_range
        res  = self.map_resolution
        size = int((2 * half) / res)

        # ---------- 1. grid–cell centres in world frame ----------
        gx = np.linspace(x - half + res/2,  x + half - res/2, size)   # shape (size,)
        gy = np.linspace(y - half + res/2,  y + half - res/2, size)   # shape (size,)
        cx, cy = np.meshgrid(gx, gy)                                  # each (size, size)

        # ---------- 2. obstacle catalogue ----------
        # static : (x, y, r)    moving : (x, y, r, vx, vy) → keep first three
        obs = np.array([
            (ox, oy, r) for (ox, oy, r) in obstacles["static"]
        ] + [
            (ox, oy, r) for (ox, oy, r, vx, vy) in obstacles["moving"]
        ])

        if obs.size == 0:
            self.grid = [[0]*size for _ in range(size)]
            return

        ox   = obs[:, 0][:, None, None]         # (N,1,1)
        oy   = obs[:, 1][:, None, None]         # (N,1,1)
        rad2 = obs[:, 2][:, None, None]**2      # squared radii  (N,1,1)

        # ---------- 3. vectorised distance test ----------
        dx2  = (cx - ox)**2                     # (N,size,size)
        dy2  = (cy - oy)**2
        inside = (dx2 + dy2) <= rad2            # boolean mask

        # ---------- 4. collapse over obstacles ----------
        occ_grid = np.any(inside, axis=0)       # (size,size)  True → occupied

        self.grid = occ_grid.astype(int).tolist()

        self.grid_np = np.array(self.grid, dtype=bool)
        self.grid_info = GridInfo(self.grid_np, half, res, size)
