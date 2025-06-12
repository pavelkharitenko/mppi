import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys, time

class DiscretePendulumEnv(gym.Env):
    """
    Discrete-time pendulum environment with Gym interface.
    Dynamics: theta_{t+1} = theta_t + theta_dot_t * dt
              theta_dot_{t+1} = theta_dot_t + (3g/(2l)*sin(theta_t) + 3u/(ml^2)) * dt
    """
    def __init__(self):
        super(DiscretePendulumEnv, self).__init__()
        
        # Physics parameters
        self.g = 9.81      # gravity (m/s^2)
        self.m = 1.0       # mass (kg)
        self.l = 1.0       # length (m)
        self.dt = 0.05     # time step (s)
        self.max_torque = 108.0  # Maximum control input (N*m)
        
        # Gym spaces
        self.action_space = spaces.Box(low=-self.max_torque, 
                                      high=self.max_torque, 
                                      shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.pi, -10]), 
                                           high=np.array([np.pi, 10]), 
                                           dtype=np.float32)
        
        # Rendering setup
        self.screen_size = 500
        self.scale = self.screen_size / 3.0  # Pixels per meter
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # State
        self.state = None
        self._sim_state = None
        
    def reset(self, *, theta=None, theta_vel=None):
        if theta is None:
            theta = np.random.uniform(-np.pi/2, np.pi/2)
        else:
            # Convert input [0,2π) to signed radians
            theta = (theta + np.pi) % (2*np.pi) - np.pi
        
        self.state = np.array([theta, theta_vel or 0.0])
        self._sim_state = self.state.copy()
        return np.array([theta % (2*np.pi), self.state[1]])  # Return [0,2π) for observation
    
    def step(self, u, simulate=False):
        # Clip torque to valid range
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        
        th, thdot = self.state
        
        # Discrete dynamics (Euler integration)
        newthdot = thdot + (3*self.g/(2*self.l)*np.sin(th) + 3*u/(self.m*self.l**2)) * self.dt
        newth = th + newthdot * self.dt
        
        # Normalize angle to [-pi, pi]
        newth = ((newth + np.pi) % (2*np.pi)) - np.pi
        
        
        # Reward function (customize for your task)
        # Goal: Swing upright (theta=0) with minimal velocity and control effort
        reward = -(np.abs(newth) + 0.1*np.abs(newthdot) + 0.01*u**2)
        
        # Termination conditions
        done = False  # Never terminates (episode length handled by wrapper)

        display_th = newth % (2*np.pi)  # For rendering and observation
        computation_th = newth  # Keep signed for physics

        if not simulate:
            self.state = np.array([computation_th, newthdot])
            return np.array([display_th, newthdot]), reward, done, {}
        else:
            self._sim_state = np.array([computation_th, newthdot])
            return np.array([display_th, newthdot]), reward, done, {}


    def vectorized_step(self, u_batch):  # u_batch shape: (M, 1)
        th, thdot = self._sim_state.T  # shape: (M,)
        newthdot = thdot + (3*self.g/(2*self.l)*np.sin(th) + 3*u_batch[:,0]/(self.m*self.l**2)) * self.dt
        newth = th + newthdot * self.dt
        self._sim_state = np.stack([newth, newthdot], axis=1)
        return self._sim_state

    def render(self, mode='human', target_theta=None, target_vel=None):
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Discrete Pendulum')
            self.clock = pygame.time.Clock()
        
        if mode == 'human':
            self.screen.fill((255, 255, 255))
            
            # Convert current state to screen coordinates
            center_x = self.screen_size // 2
            center_y = self.screen_size // 2
            th, thdot = self.state
            
            # Draw target position (if provided)
            if target_theta is not None:
                # Convert target angle to screen coordinates
                target_x = center_x + self.l * self.scale * np.sin(target_theta)
                target_y = center_y - self.l * self.scale * np.cos(target_theta)
                
                # Draw dashed grey line for target
                pygame.draw.line(self.screen, (200, 200, 200), 
                            (center_x, center_y), (target_x, target_y), 2)
                
                # Draw grey circle at target position
                pygame.draw.circle(self.screen, (200, 200, 200), 
                                (int(target_x), int(target_y)), 10)
                
                # Render target text
                font = pygame.font.SysFont('Arial', 16)
                theta_text = f"Target θ: {target_theta:.2f} rad"
                if target_vel is not None:
                    theta_text += f", ω: {target_vel:.2f} rad/s"
                
                text_surface = font.render(theta_text, True, (100, 100, 100))
                self.screen.blit(text_surface, (10, 10))
            
            # Draw current pendulum (original code)
            bob_x = center_x + self.l * self.scale * np.sin(th)
            bob_y = center_y - self.l * self.scale * np.cos(th)
            pygame.draw.line(self.screen, (0, 0, 0), 
                            (center_x, center_y), (bob_x, bob_y), 3)
            pygame.draw.circle(self.screen, (200, 0, 0), 
                            (int(bob_x), int(bob_y)), 15)
            
            # Render current state text
            font = pygame.font.SysFont('Arial', 16)
            state_text = f"Current θ: {th:.2f} rad, ω: {thdot:.2f} rad/s"
            text_surface = font.render(state_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 30))
            
            pygame.display.flip()
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return   
 
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
            self.screen = None

# =============================================
# Example Usage: Random Agent
# =============================================
if __name__ == "__main__":
    env = DiscretePendulumEnv()
    obs = env.reset()
    
    for _ in range(500):
        action = env.action_space.sample()  # Random torque
        obs, reward, done, _ = env.step(action)
        env.render()
        print(obs)
        #time.sleep(0.2)
        if done:
            obs = env.reset()
    
    env.close()