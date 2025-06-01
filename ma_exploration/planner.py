import math

class Planner:
    def __init__(self, goal, sensor_range, map_resolution=0.25):
        self.goal = goal  # (x_goal, y_goal)
        self.sensor_range = sensor_range  # in meters
        self.map_resolution = map_resolution
        self.grid = None

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
        size = int((2 * self.sensor_range) / self.map_resolution)
        grid = [[0 for _ in range(size)] for _ in range(size)]
        half_range = self.sensor_range
        resolution = self.map_resolution

        for ox, oy, r in obstacles["static"]:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy)

            if dist <= self.sensor_range + r:
                min_x = ox - r
                max_x = ox + r
                min_y = oy - r
                max_y = oy + r

                gx_min = int((min_x - (x - half_range)) / resolution)
                gx_max = int((max_x - (x - half_range)) / resolution) + 1
                gy_min = int((min_y - (y - half_range)) / resolution)
                gy_max = int((max_y - (y - half_range)) / resolution) + 1

                for gx in range(gx_min, gx_max):
                    for gy in range(gy_min, gy_max):
                        if 0 <= gx < size and 0 <= gy < size:
                            cx = x - half_range + gx * resolution + resolution / 2
                            cy = y - half_range + gy * resolution + resolution / 2
                            # Use a tighter threshold by expanding cell size as an area patch
                            corners = [
                                (cx - resolution / 2, cy - resolution / 2),
                                (cx + resolution / 2, cy - resolution / 2),
                                (cx - resolution / 2, cy + resolution / 2),
                                (cx + resolution / 2, cy + resolution / 2)
                            ]
                            if any(math.hypot(px - ox, py - oy) <= r for px, py in corners):
                                grid[gy][gx] = 1

        self.grid = grid