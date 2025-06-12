import math
import numpy as np

class DifferentialDriveDynamics:
    """
    Differential Drive Dynamics (discrete-time)

    State: x = [x, y, theta]
    Control: u = [v, w] (linear velocity, angular velocity)

    Update:
        x_next = x + v * cos(theta) * dt
        y_next = y + v * sin(theta) * dt
        theta_next = theta + w * dt
    """
    def __init__(self):
        self.state_dim = 3
        self.control_dim = 2

    def step(self, state, control, dt):
        #print(control)
        x, y, theta = state      # state: position (x, y) and orientation (theta)
        v, w = control           # control: linear velocity (v), angular velocity (w)
        #dt = self.dt

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt

        return x, y, theta

    def step_vectorized(self, states, controls, dt):
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        
        v = controls[:, 0]
        w = controls[:, 1]

        x_next = x + np.cos(theta) * dt
        y_next = y + np.sin(theta) * dt

        theta_next = theta + w * dt

        # if needed: wrap theta between [-pi, pi]
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi

        return np.stack([x_next, y_next, theta_next], axis=1)


class Holonomic2DDynamics:
    """
    Holonomic 2D Dynamics (discrete-time)

    State: x = [x, y, theta]
    Control: u = [vx, vy] (velocity in x and y direction)

    Update:
        x_next = x + vx * dt
        y_next = y + vy * dt
        theta is ignored or set to zero
    """
    #def __init__(self, dt):
    #    self.dt = dt

    def step(self, state, control, dt):
        x, y, _ = state          # state: position (x, y), orientation (ignored)
        vx, vy = control         # control: velocity in x and y directions
        #dt = self.dt

        x += vx * dt
        y += vy * dt
        theta = 0.0

        return x, y, theta