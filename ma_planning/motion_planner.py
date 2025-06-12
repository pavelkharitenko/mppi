from controller import *
from pygame_env_utils import *

def generate_circle_positions(num_robots, center, radius):
    """Returns a list of (x, y, theta) tuples spaced evenly on a circle."""
    positions = []
    for i in range(num_robots):
        angle = 2 * np.pi * i / num_robots
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        theta = angle + np.pi  # Facing inward
        positions.append((x, y, theta))
    return positions

def assign_antipodal_goals(positions):
    """Returns a list of (goal_x, goal_y) pairs based on antipodal assignment."""
    num = len(positions)
    goals = []
    for i in range(num):
        antipodal_idx = (i + num // 2) % num
        gx, gy, _ = positions[antipodal_idx]
        goals.append((gx, gy))
    return goals

def create_robots_with_goals(num_robots, center, radius, dynamics_model, use_single_agent_controller=True):
    """
    Creates and returns a list of Robot instances with antipodal goals.
    
    """
    start_positions = generate_circle_positions(num_robots, center, radius)
    goal_positions = assign_antipodal_goals(start_positions)
    start_and_goals = list(zip(start_positions, goal_positions))

    robots = []
    for i in range(num_robots):
        if use_single_agent_controller:
            controller = MPPIController(
                model=dynamics_model,
                horizon=150,
                stepsize=0.02,
                num_samples=1000,
                lambda_=0.01,
                noise_std=np.array([0.1,0.02]))
        else:
            controller = None  # MAMPPI will be used instead

        (x, y, theta), (gx, gy) = start_and_goals[i]
        robot = Robot(
            x, y, theta,
            goal=(gx, gy),
            sensor_range=10.0,
            dynamics_model=dynamics_model,
            controller=controller,
            local_planning=True
        )
        robots.append(robot)




    #i = 0

    #    i += 1
    return robots


class D4ormPlanner():

    def __init__(self, num_robots, start_states, goal_states, model, horizon, stepsize, num_samples, lambda_, noise_std):
        self.num_robots = num_robots
        self.start_states = start_states
        self.goal_states = goal_states
        self.model = model # dynamics
        self.horizon = horizon # K stepsteps per rollout
        self.num_samples = num_samples # M parallel rollouts
        self.lambda_ = lambda_
        self.dt = stepsize
        self.noise_std = noise_std

        # nominal control trajectory
        self.U = np.zeros((self.num_robots, self.horizon, self.model.control_dim)) # start with zero-mean control traj.
        self.U[:,:, 0] = 0.6


        # buffers for evaluation
        self.U_noisy = np.zeros((self.num_robots, self.horizon, self.model.control_dim))
        self.state_rollouts = np.zeros((self.num_samples, self.horizon+1, self.model.state_dim))

        # results
        self.control_sequences = self.U
        # optimal rollout
        self.optimal_rollouts = np.zeros((self.num_robots, self.horizon + 1, self.model.state_dim))
    
    def get_plan(self):
        self.plan_optimal_trajectories()
        return self.control_sequences, self.optimal_rollouts
    

    def plan_optimal_trajectories(self):


        # sample K x M control inputs
        u_noise = np.random.normal(0, self.noise_std, self.U.shape)
        self.U_noisy = self.U + u_noise

        self.control_sequences = self.U_noisy

        #print("U-noisy", self.U_noisy)

        # store optimal rollout for debugging
        state = self.start_states
        self.optimal_rollouts[:, 0, :] = state

        for k in range(self.horizon):
            U_k = self.U_noisy[:,k,:]
            state = self.model.step_vectorized(state, U_k, self.dt)
            self.optimal_rollouts[:, k+1, :] = state


