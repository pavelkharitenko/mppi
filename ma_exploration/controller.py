import math
import numpy as np
from plots_utils import *
import matplotlib.pyplot as plt

class BaseController:
    def compute_action(self, state, local_goal, occupancy_grid):
        raise NotImplementedError

class MPPIController(BaseController):
    """
    Single-agent Model Predictive Path Integral Controller
    """
    def __init__(self, model, horizon, stepsize, num_samples, lambda_, noise_std):
        self.model = model
        self.horizon = horizon # K stepsteps per rollout
        self.num_samples = num_samples # M parallel rollouts
        self.lambda_ = lambda_
        self.dt = stepsize
        self.noise_std = noise_std

        # nominal control trajectory
        self.U = np.zeros((self.horizon, self.model.control_dim)) # start with zero-mean control traj.
        self.U[:, 0] = 0.3

        # optimal rollout
        self.optimal_rollout = np.zeros((1, self.horizon + 1, self.model.state_dim))

        # buffers for evaluation
        self.U_noisy = np.zeros((self.num_samples, self.horizon, self.model.control_dim))
        self.state_rollouts = np.zeros((self.num_samples, self.horizon+1, self.model.state_dim))

    def compute_action(self, state, local_goal, occupancy_grid):
        v, w = 0, 0

        # sample K x M control inputs
        u_noise = np.random.normal(0, self.noise_std, (self.num_samples, self.horizon, self.model.control_dim))
        self.U_noisy = self.U[np.newaxis, :, :] + u_noise

        # perform rollouts in parallel and get costs
        self.rollout(state)
        costs = self.cost_function(local_goal, occupancy_grid)
        #print(costs)


        # compute weights, update nominal control sequence, return first input
        weights = np.exp(-(costs - np.min(costs)) / self.lambda_)
        #plt.hist(weights, bins=50); plt.show()
        weights /= np.sum(weights)
        self.U = np.sum(weights[:, np.newaxis, np.newaxis] * self.U_noisy, axis=0)

        # store optimal rollout for debugging
        self.optimal_rollout[0, 0] = state

        for k in range(self.horizon):
            u_k = self.U[k]  # shape (control_dim,)
            state = self.model.step(state, u_k, self.dt)  # shape (state_dim,)
            self.optimal_rollout[0, k + 1] = state


        #print(self.U[0,:])
        #plot_u_histograms(self.U_noisy, costs, self.lambda_, control_names=["v", "w"])
        
        return self.U[0,:]
    
    def rollout(self, state):
        state = np.tile(state, (self.num_samples, 1))
        self.state_rollouts[:, 0, :] = state
        for k in range(self.horizon):
            U_k = self.U_noisy[:,k,:]
            state = self.model.step_vectorized(state, U_k, self.dt)
            self.state_rollouts[:, k+1, :] = state
    
    def cost_function(self, local_goal, occupancy_grid):
        #costs = np.zeros(self.num_samples)
        
        # compute control input cost
        cost_sample = 0.001 * np.sum(self.U_noisy**2, axis=(1, 2))

        # compute tracking costs


        positions = self.state_rollouts[:, 1:, :2]
        goal_diff = positions - local_goal
        l2_dist = np.linalg.norm(goal_diff, axis=-1)

        

        initial_states = self.state_rollouts[:, 0, :2] # first state is [x, y, theta]
        final_states = self.state_rollouts[:, -1, :2]  # final state is [x, y, theta]
        initial_goal_dist = np.linalg.norm(initial_states - local_goal, axis=1)
        final_dist = np.linalg.norm(final_states - local_goal, axis=1)

        cost_tracking = np.sum(l2_dist, axis=-1)

        # cost obstacle
        # TODO

        #plt.plot(self.state_rollouts[:, :, 0], self.state_rollouts[:, :, 1], 'r-', alpha=0.1)
        #plt.plot(local_goal[0], local_goal[1], 'g*')
        #plt.show()

        #plt.plot(self.state_rollouts[0, :, 0], self.state_rollouts[0, :, 1], 'r-', alpha=0.1)
        #plt.plot(local_goal[0], local_goal[1], 'g*')
        #plt.show()

        return cost_tracking #+ cost_sample 




class GlobalController(BaseController):
    """
    Moves agent toward global goal, no local planning
    """
    def compute_action(self, state, goal, occupancy_grid):
        x, y, theta = state
        tx, ty = goal
        angle_to_target = math.atan2(ty - y, tx - x)
        angle_diff = angle_to_target - theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        v = 1.0
        w = 2.0 * angle_diff
        return v, w