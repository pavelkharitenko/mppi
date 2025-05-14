import matplotlib.pyplot as plt
from pendulum_env import DiscretePendulumEnv
from mppi import MPPI
import numpy as np
import time

env = DiscretePendulumEnv()

target_theta = 0.5*np.pi
target_vel = 0.0
mppi = MPPI(env, K=20, M=3000, noise_std=1.5, target_theta=target_theta, target_vel=target_vel)

state = env.reset(theta=np.pi, theta_vel=0.0)
states = [state.copy()]
debug_rollouts = []  # To store debug information

stop = True

for i in range(200):
    action = mppi.update(state)
    state, _, _, _ = env.step(action)
    states.append(state.copy())
    env.render(target_theta=target_theta, target_vel=target_vel)
    
    if stop:

        time.sleep(14)
        stop = False
    
    if i % 10 == 0:
        debug_rollouts.append({
            'step': i,
            'current_state': state.copy(),
            'U': mppi.U.copy(),
            'next_action': action
        })

# Convert to numpy array
states = np.array(states)

# Create figure
plt.figure(figsize=(12, 8))

# Main plot with π scaling
plt.subplot(2, 2, 1)
plt.plot(states[:, 0]/np.pi, label='Theta')  # Scale by π
plt.plot(states[:, 1], label='Theta_dot (rad/s)')
plt.axhline(target_theta/np.pi, color='r', linestyle='--', label='Target Theta')
plt.axhline(target_vel, color='g', linestyle='--', label='Target Vel')
plt.title('State Trajectory (θ in π radians)')
plt.ylabel('θ [π rad] / θ̇ [rad/s]')
plt.legend()
plt.grid(True)

# Customize y-axis ticks for theta (in π units)
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda val, pos: f'{val:.1f}π' if val != 0 else '0'
))

# Plot actions over time
plt.subplot(2, 2, 2)
actions = [mppi.U[0][0] for _ in range(len(states)-1)]
plt.plot(actions)
plt.title('Applied Actions Over Time')
plt.xlabel('Time step')
plt.ylabel('Control torque')
plt.grid(True)

# Debug plot - control sequences
plt.subplot(2, 2, 3)
for i, debug in enumerate(debug_rollouts[:5]):
    plt.plot(debug['U'], label=f'Step {debug["step"]}')
plt.title('Optimal Control Sequences')
plt.xlabel('Horizon step')
plt.ylabel('Control value')
plt.legend()
plt.grid(True)

# Debug plot - trajectories with π scaling
plt.subplot(2, 2, 4)
for i, debug in enumerate(debug_rollouts[:3]):
    env._sim_state = debug['current_state'].copy()
    traj = [debug['current_state'].copy()]
    for u in debug['U']:
        next_state = env.step(u, simulate=True)[0]
        traj.append(next_state.copy())
        env._sim_state = next_state
    traj = np.array(traj)
    plt.plot(traj[:, 0]/np.pi, label=f'Step {debug["step"]} θ')
    plt.plot(traj[:, 1], '--', label=f'Step {debug["step"]} θ̇')
plt.title('Predicted Trajectories (θ in π radians)')
plt.xlabel('Horizon step')
plt.ylabel('θ [π rad] / θ̇ [rad/s]')
plt.legend()
plt.grid(True)

# Apply π scaling to the right subplot
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda val, pos: f'{val:.1f}π' if val != 0 else '0'
))

plt.tight_layout()
plt.show()