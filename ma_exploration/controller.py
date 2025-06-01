import math


class BaseController:
    def compute_action(self, state, local_goal, occupancy_grid):
        raise NotImplementedError

class MPPIController(BaseController):
    def __init__(self, model, horizon, num_samples, lambda_):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_

    def compute_action(self, state, local_goal, occupancy_grid):
        # Implement MPPI optimization here
        # Return (v, w)
        pass


class GlobalController(BaseController):
    
    def compute_action(self, state, goal, occupancy_grid):
        x, y, theta = state
        tx, ty = goal
        angle_to_target = math.atan2(ty - y, tx - x)
        angle_diff = angle_to_target - theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        v = 1.0
        w = 2.0 * angle_diff

        return v, w