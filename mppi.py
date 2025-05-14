import numpy as np
from pendulum_env import DiscretePendulumEnv

class MPPI:
    def __init__(self, env, K=20, M=30, lambda_=1.0, noise_std=0.5, target_theta=0.0, target_vel=0.0):
        self.env = env
        self.K = K                  # Horizon length
        self.M = M                  # Number of trajectories
        self.lambda_ = lambda_      # Temperature parameter
        self.noise_std = noise_std  # Std dev of control noise
        self.target_theta = target_theta
        self.target_vel = target_vel
        
        # Action sequence (shape: K x 1)
        self.U = np.zeros((self.K, 1))
        self._sim_state = None  # Placeholder for MPPI rollouts

    def cost_fn(self, state, action):
        """Convert observed [0,2π) state back to signed radians for cost calculation"""
        theta_obs, theta_dot = state
        theta_comp = (theta_obs + np.pi) % (2*np.pi) - np.pi  # Convert to signed
        
        # cost is current - target position
        cost_theta = (theta_comp - self.target_theta)**2
        cost_vel = self.target_vel**2
        cost_action = 0.001 * action**2
        return cost_theta + cost_vel + cost_action
    
    def vectorized_cost_fn(self, state, action):
        theta_obs, theta_dot = state[:,0], state[:,1]
        theta_comp = (theta_obs + np.pi) % (2*np.pi) - np.pi 

        # Ensure all operations are element-wise
        cost_theta = (theta_comp - self.target_theta)**2          # shape (M,)
        cost_vel = (theta_dot - self.target_vel)**2               # shape (M,) 
        cost_action = 0.001 * action[:,0]**2 if action.ndim > 1 else 0.001 * action**2  # handle both (M,) and (M,1)
        
        return cost_theta + cost_vel + cost_action  # shape (M,)


    def rollout(self, state, U_noisy):
        """Simulate trajectory with noisy controls"""
        total_cost = 0
        self.env._sim_state = state.copy()  # Initialize simulated state
        
        for k in range(self.K):
            u = U_noisy[k]
            # Use simulate=True and get next state
            next_state = self.env.step(u, simulate=True)[0]  # [0] gets state from (state, reward, done, info)
            total_cost += self.cost_fn(next_state, u)
            self.env._sim_state = next_state  # Update for next step
        
        return total_cost
    
    def rollout_batch(self, state, U_noisy_batch):
        batch_size = U_noisy_batch.shape[0]
        costs = np.zeros(batch_size)
        self.env._sim_state = np.tile(state, (batch_size, 1)) # init all trajectories

        for k in range(self.K):
            # Vectorized step (all m rollouts simultaneously)
            next_states = self.env.vectorized_step(U_noisy_batch[:,k])
            costs += self.vectorized_cost_fn(next_states, U_noisy_batch[:,k])
            self.env._sim_state = next_states
        
        return costs


 
    def update(self, state):
        # Generate M noisy action sequences (shape: M x K x 1)
        noise = np.random.normal(0, self.noise_std, (self.M, self.K, 1))
        U_noisy = self.U + noise
        
        # Evaluate all trajectories
        costs = self.rollout_batch(state, U_noisy)
        self.current_costs = costs  # Store for debugging
        
        # Compute weights (lower cost => higher weight)
        weights = np.exp(-self.lambda_ * (costs - np.min(costs)))
        weights /= np.sum(weights)  # Normalize
        self.current_weights = weights  # Store for debugging
        
        # Update nominal control sequence
        self.U = np.sum(weights[:, None, None] * U_noisy, axis=0)
        
        return self.U[0]  # Return first action
    
    def get_control_sequence(self):
        """Return the current optimal control sequence"""
        return self.U.copy()
    
    def reset(self):
        self.U.fill(0)


