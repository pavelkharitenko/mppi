import math
import numpy as np
from plots_utils import *
from planner import GridInfo
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
        self.U[:, 0] = 0.45

        # optimal rollout
        self.optimal_rollout = np.zeros((1, self.horizon + 1, self.model.state_dim))

        # buffers for evaluation
        self.U_noisy = np.zeros((self.num_samples, self.horizon, self.model.control_dim))
        self.state_rollouts = np.zeros((self.num_samples, self.horizon+1, self.model.state_dim))

    def compute_action(self, state, local_goal, grid_info):
        v, w = 0, 0

        state = np.array(state)
        self.state = state

        # sample K x M control inputs
        u_noise = np.random.normal(0, self.noise_std, (self.num_samples, self.horizon, self.model.control_dim))
        self.U_noisy = self.U[np.newaxis, :, :] + u_noise

        # perform rollouts in parallel and get costs
        self.rollout(state)
        costs = self.cost_function(local_goal, grid_info)


        # compute weights, update nominal control sequence, return first input
        weights = np.exp(-(costs - np.min(costs)) / self.lambda_)
        
        weights /= np.sum(weights)
        self.U = np.sum(weights[:, np.newaxis, np.newaxis] * self.U_noisy, axis=0)

        # store optimal rollout for debugging
        self.optimal_rollout[0, 0] = state

        for k in range(self.horizon):
            u_k = self.U[k]  # shape (control_dim,)
            #state = self.model.step(state, u_k, self.dt)  # shape (state_dim,)
            state = np.array(self.model.step_vectorized(state[np.newaxis, :], u_k[np.newaxis, :], self.dt)[0])
            self.optimal_rollout[0, k + 1] = state

        #plot_u_histograms(self.U_noisy, costs, self.lambda_, control_names=["v", "w"])
        #plot_rollouts_and_hist(self.state_rollouts, weights, goal=local_goal, optimal_rollout=self.optimal_rollout)
        #plot_cost_weight_debug(costs, weights, self.lambda_)
        #plot_rollouts_with_collision(self.state_rollouts, weights, collided_mask=self.collided_mask,goal=local_goal, optimal_rollout=self.optimal_rollout)
        
        return self.U[0,:]
    
    def rollout(self, state):
        state = np.tile(state, (self.num_samples, 1))
        self.state_rollouts[:, 0, :] = state
        for k in range(self.horizon):
            U_k = self.U_noisy[:,k,:]
            state = self.model.step_vectorized(state, U_k, self.dt)
            self.state_rollouts[:, k+1, :] = state
    
    def cost_function(self, local_goal, grid_info: GridInfo):
        
        # compute control effort cost
        sigma_inv   = 1.0 / (self.noise_std**2)
        u_nom = self.U[np.newaxis, :, :]
        u_delta = self.U_noisy - u_nom
        u_cost = np.sum(u_nom * sigma_inv * u_nom, axis=-1)
        u_udelta_cost = 2.0 * np.sum(u_nom * sigma_inv * u_delta, axis=-1)

        cost_sample = (0.5/self.lambda_) * np.sum(u_cost + u_udelta_cost,axis=-1)
        #cost_sample = 0.01 * np.sum(self.U_noisy**2, axis=(1, 2))

        # compute tracking costs
        positions = self.state_rollouts[:, 1:, :2] # get (x,y) pos. of new trajectories
        goal_diff = positions - local_goal # each x,y pos. 
        l2_dist = np.linalg.norm(goal_diff, axis=-1) # compute || x_i - x_d ||
        cost_tracking = np.sum(l2_dist, axis=-1)

        #initial_states = self.state_rollouts[:, 0, :2] # first state is [x, y, theta]
        #final_states = self.state_rollouts[:, -1, :2]  # final state is [x, y, theta]

        # cost obstacle
        occ = grid_info.occ
        half = grid_info.half
        res = grid_info.res 
        S = grid_info.S

        gx = ((positions[:, :, 0] - (self.state[0] - half)) / res).astype(int)
        gy = ((positions[:, :, 1] - (self.state[1] - half)) / res).astype(int)
        gx = np.clip(gx, 0 , S-1)
        gy = np.clip(gy, 0 , S-1)

        collided = occ[gy, gx]
        has_collision = np.any(collided, axis=1)
        self.collided_mask = has_collision
        
        cost_collision = 2e3 * has_collision.astype(float)
        

        cost_total = cost_collision + cost_tracking + 10* cost_sample#0.2 * cost_tracking #17.0 * cost_sample #+ 0.3 * cost_tracking + cost_collision

        self.cost_total = cost_total

        #plot_cost_components(10* cost_sample, cost_tracking, cost_collision, cost_total=cost_total)

        #plt.plot(self.state_rollouts[:, :, 0], self.state_rollouts[:, :, 1], 'r-', alpha=0.1)
        #plt.plot(local_goal[0], local_goal[1], 'g*')
        #plt.show()

        return cost_total
    


class D4ormSAController(BaseController):
    """
    Single-agent D4orm (Model-based Diffusion) Controller
    """
    def __init__(self, model, horizon, stepsize, num_samples, diffusion_steps, lambda_, alpha_decay):
        self.model = model
        self.horizon = horizon # K stepsteps per rollout
        self.num_samples = num_samples # M parallel rollouts
        self.diffusion_steps = diffusion_steps
        self.lambda_ = lambda_
        self.dt = stepsize
        self.alpha_decay = alpha_decay

        # nominal control trajectory
        self.U = np.zeros((self.horizon, self.model.control_dim)) # start with zero-mean control traj.
        self.U[:, 0] = 0.45

        # optimal rollout
        self.optimal_rollout = np.zeros((1, self.horizon + 1, self.model.state_dim))

        # buffers for evaluation
        self.U_noisy = np.zeros((self.num_samples, self.horizon, self.model.control_dim))
        self.state_rollouts = np.zeros((self.num_samples, self.horizon+1, self.model.state_dim))

    def compute_action(self, state, local_goal, grid_info):
        state = np.array(state)
        self.state = state


        #self.U = np.zeros((self.horizon, self.model.control_dim)) # start with zero-mean control traj.

        # Set up noise scale schedule (can be linear or cosine)
        alpha_schedule = np.linspace(0.95, 0.6, self.diffusion_steps)  # from high to lower certainty
        alpha_bar = np.cumprod(alpha_schedule)  # ᾱ_t = Π α_s

        # for i,...,N do backward diffusion steps

        for i in range(self.diffusion_steps):

            a_bar = alpha_bar[i]
            noise_std = np.sqrt((1 / a_bar - 1))
            #noise_std = noise_std.reshape(1, self.horizon, 1)

            # Sample control noise: ε ~ N(0, σ²)
            noise = np.random.normal(0, noise_std, (self.num_samples, self.horizon, self.model.control_dim))


            # Compute Gamma_i ~ N(U / sqrt(ᾱ), σ² / ᾱ)
            mean = self.U[np.newaxis, :, :] / np.sqrt(a_bar)
            scaled_noise = noise / np.sqrt(a_bar)
            Gamma = mean + scaled_noise
            self.U_noisy = Gamma


            #self.U_noisy = self.U[np.newaxis, :, :] + u_noise / self.alpha_i

            # perform rollouts in parallel
            self.rollout(state)
            rewards = self.reward_function(local_goal, grid_info)



            # compute weights, update nominal control sequence, return first input
            rewards_scaled = rewards - np.max(rewards) / self.lambda_
            weights = np.exp(rewards_scaled)
        
            weights /= np.sum(weights) # normalize
            # Monte Carlo estimate of U_bar
            U_bar = np.sum(weights[:, np.newaxis, np.newaxis] * self.U_noisy, axis=0)


            # Denoising step: scale back to get U_{i-1}
            if i < self.diffusion_steps -1:
                self.U = np.sqrt(alpha_bar[i+1]) * U_bar

            #self.alpha_i *= self.alpha_decay

        # store optimal rollout for debugging
        self.optimal_rollout[0, 0] = state

        for k in range(self.horizon):
            u_k = self.U[k]  # shape (control_dim,)
            #state = self.model.step(state, u_k, self.dt)  # shape (state_dim,)
            state = np.array(self.model.step_vectorized(state[np.newaxis, :], u_k[np.newaxis, :], self.dt)[0])
            self.optimal_rollout[0, k + 1] = state

        #plot_u_histograms(self.U_noisy, costs, self.lambda_, control_names=["v", "w"])
        plot_rollouts_and_hist(self.state_rollouts, weights, goal=local_goal, optimal_rollout=self.optimal_rollout)
        #plot_cost_weight_debug(costs, weights, self.lambda_)
        #plot_rollouts_with_collision(self.state_rollouts, weights, collided_mask=self.collided_mask,goal=local_goal, optimal_rollout=self.optimal_rollout)
        
        return self.U[0,:]
    
    def rollout(self, state):
        state = np.tile(state, (self.num_samples, 1))
        self.state_rollouts[:, 0, :] = state

        for k in range(self.horizon):
            U_k = self.U_noisy[:, k, :]
            state = self.model.step_vectorized(state, U_k, self.dt)
            self.state_rollouts[:, k+1, :] = state
    
    def reward_function(self, local_goal, grid_info: GridInfo):
        local_goal = np.array(local_goal)
        
        # compute control effort cost
        #sigma_inv   = 1.0 / (self.noise_std**2)
        #u_nom = self.U[np.newaxis, :, :]
        #u_delta = self.U_noisy - u_nom
        #u_cost = np.sum(u_nom * sigma_inv * u_nom, axis=-1)
        #u_udelta_cost = 2.0 * np.sum(u_nom * sigma_inv * u_delta, axis=-1)

        #cost_sample = (0.5/self.lambda_) * np.sum(u_cost + u_udelta_cost,axis=-1)
        #cost_sample = 0.01 * np.sum(self.U_noisy**2, axis=(1, 2))

        # reward goal

        post_start = self.state_rollouts[:, 1,:2] # shape (N, 1, 2) position at t=1
        pos_t = self.state_rollouts[:, 1:, :2] # (N, T, 2) positions at times t > 1

        goal = local_goal[np.newaxis, np.newaxis, :] # (1, 1, 2)
        dist_to_goal = np.linalg.norm(pos_t - goal, axis=-1) # (N, 2)

        dist_total = np.linalg.norm(post_start - local_goal, axis=-1, keepdims=True) # (N, 1)
        dist_total = np.where(dist_total < 1e-6, 1e-6, dist_total) # for numeric stability, dont divide by smaller than 1e-6

        reward_goal = 1.0 - (dist_to_goal / dist_total) # equation (4)

        

        # reward safety/collision

        # TODO collision with other trajectories

        # TODO collision with obstacles 

        

        self.reward_total = reward_goal.sum(axis=1) / self.horizon

        #plot_cost_components(10* cost_sample, cost_tracking, cost_collision, cost_total=cost_total)

        #plt.plot(self.state_rollouts[:, :, 0], self.state_rollouts[:, :, 1], 'r-', alpha=0.1)
        #plt.plot(local_goal[0], local_goal[1], 'g*')
        #plt.show()

        return self.reward_total
    




class MAMPPIController(BaseController):
    """
    Multi-Agent MPPI controller
    """
    def __init__(self, model, horizon, stepsize, num_samples, lambda_, noise_std, num_robots):
        self.model = model
        self.horizon = horizon
        self.dt = stepsize
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_std = noise_std
        self.num_robots = num_robots

        self.U = np.zeros((self.num_robots, self.horizon, self.model.control_dim))
        self.U[...,0] = 0.6

        self.optimal_rollouts = np.zeros((self.num_robots, self.horizon+1, self.model.state_dim))

    
    def compute_actions(self, states, local_goals, occupancy_grids):
        
        
        return self.U[:,0]







class TrackingController(BaseController):
    """
    Executes a preplanned sequence of control actions (e.g., from a planner).
    Assumes the plan is stored in self.optimal_rollout and consumed sequentially.
    """

    def __init__(self):
        self.control_trajectory = []
        self.optimal_rollout = []  # List of (v, w) tuples or np.ndarray
        self.step_index = 0
        self.control_dim = None
        self.state_dim = None 

    def set_plan(self, control_sequence, optimal_rollout):
        """
        Set a new plan to follow.
        control_sequence: list or np.ndarray of shape (T, control_dim)
        """
        self.control_trajectory = control_sequence
        self.optimal_rollout = optimal_rollout
        #print(self.optimal_rollout)
        self.step_index = 0
        self.control_dim = control_sequence.shape[-1]
        

    def compute_action(self, state, goal, occupancy_grid):
        """
        Returns the next control action from the planned sequence.
        If finished, returns zero velocity.
        """
        if self.step_index < len(self.control_trajectory):
            u = self.control_trajectory[self.step_index]
            self.step_index += 1
            return tuple(u)
        else:
            print("plan is exhausted")
            return np.zeros(shape=self.control_dim)  # Stop if plan is exhausted


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
    

