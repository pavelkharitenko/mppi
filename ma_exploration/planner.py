import math

class Planner:
    def __init__(self, goal, sensor_range):
        self.goal = goal  # (x_goal, y_goal)
        self.sensor_range = sensor_range  # in meters

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