import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as torch_nn
import torch.optim as optim
import torch.nn.functional as F

# Fix for VS Code to save plots as images
matplotlib.use('Agg')

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUSTOM MEC ENVIRONMENT
class MECOffloadingEnv:
    def __init__(self):
        # Action Space: 0 = Local, 1 = Cloud, 2 = MEC
        self.action_space_n = 3
        # State Space: [i(t), c(t), d(t), Re(t), a_t-1]
        self.observation_space_n = 5
        
        # Suggested Parameters
        self.f_l = 1.5e9      # Device CPU frequency
        self.f_e = 10e9       # MEC CPU frequency
        self.f_c = 50e9       # Cloud CPU frequency
        self.P_tx = 0.5       # Device transmission power (W)
        self.T_bh = 0.2       # Backhaul propagation delay (s)
        self.b_bh = 100e6     # Backhaul bandwidth
        
        # Energy coefficients
        self.zeta = 1e-27     # Hardware coefficient
        self.e_e = 5e-10      # MEC energy per cycle
        self.e_c = 2e-10      # Cloud energy per cycle
        self.P_bh = 0.2       # Backhaul transmission power
        
        # Reward parameters
        self.lam = 10.0       # Penalty for deadline violation
        
        # Environment state tracking
        self.current_step = 0
        self.max_steps_per_episode = 50 # 1 episode
        self.last_action = 0  # a_{t-1}

    def reset(self):
        "Resets the environment for a new episode."
        self.current_step = 0
        self.last_action = 0
        return self._generate_state()

    def _generate_state(self):
        "Generates random task parameters for the current time slot."
        # Randomize values slightly around the PDF defaults to make the agent learn
        i_t = random.uniform(10e6, 20e6)   # Input size bits (around 16e6)
        c_t = random.uniform(0.8e9, 1.5e9) # CPU cycles (around 1.2e9)
        d_t = random.uniform(0.5, 1.5)     # Deadline in seconds (around 1.0)
        Re_t = random.uniform(15e6, 25e6)  # Uplink rate (around 20e6)
        
        # State: [i(t), c(t), d(t), Re(t), a_t-1]
        state = np.array([i_t, c_t, d_t, Re_t, self.last_action], dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def step(self, action, state):
        "Executes the chosen action and calculates Energy, Delay, and Reward."
        self.current_step += 1
        
        # Extract variables from state tensor
        i_t = state[0][0].item()
        c_t = state[0][1].item()
        d_t = state[0][2].item()
        Re_t = state[0][3].item()
        
        T_t, E_t = 0.0, 0.0
        
        # Action 0: LOCAL
        if action == 0:
            T_t = c_t / self.f_l
            E_t = c_t * self.zeta * (self.f_l ** 2)
            
        # Action 1: CLOUD 
        elif action == 1:
            T_t = (i_t / Re_t) + self.T_bh + (c_t / self.f_c)
            E_t = (self.P_tx * (i_t / Re_t)) + (self.P_bh * (i_t / self.b_bh)) + (c_t * self.e_c)
            
        # Action 2: MEC 
        elif action == 2:
            T_t = (i_t / Re_t) + (c_t / self.f_e)
            E_t = (self.P_tx * (i_t / Re_t)) + (c_t * self.e_e)
            
        # Reward Calculation 
        # rt = -[E(t)] - lambda * max(0, T(t) - d(t))
        deadline_penalty = max(0.0, T_t - d_t)
        reward_val = -E_t - (self.lam * deadline_penalty)
        reward = torch.tensor([reward_val], device=device)
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps_per_episode)
        
        # Generate next state
        self.last_action = action
        next_state = None if done else self._generate_state()
        
        return next_state, reward, done

# DQN NEURAL NETWORK & REPLAY BUFFER
class DQN(torch_nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 3 layers to find patterns between State (5 numbers) and Actions (3 numbers)
        self.layer1 = torch_nn.Linear(n_observations, 128)
        self.layer2 = torch_nn.Linear(128, 128)
        self.layer3 = torch_nn.Linear(128, n_actions)

    def forward(self, x):
        # We use ReLU activation functions for hidden layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Replay buffer to store memories
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# TRAINING HYPERPARAMETERS
BATCH_SIZE = 128
GAMMA = 0.99           # Discount factor
EPS_START = 0.9        # 90% chance to explore randomly at start
EPS_END = 0.05         # 5% chance to explore at end
EPS_DECAY = 1000       # How fast to shift from exploration to exploitation
TAU = 0.005            # Update rate for target network
LR = 1e-4              # Learning rate

env = MECOffloadingEnv()

# Create main network (policy_net) and backup network (target_net)
policy_net = DQN(env.observation_space_n, env.action_space_n).to(device)
target_net = DQN(env.observation_space_n, env.action_space_n).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # Epsilon greedy strategy calculation
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # Exploitation: pick action with highest predicted Q-value
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Exploration: pick random action (0, 1, or 2)
        return torch.tensor([[random.randrange(env.action_space_n)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare batches
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    if len(non_final_next_states_list) == 0:
        return  # Skip this optimization step if the batch is completely empty
    non_final_next_states = torch.cat(non_final_next_states_list)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Expected Q values (Bellman Equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss (Huber loss)
    criterion = torch_nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# MAIN TRAINING LOOP
num_episodes = 300
episode_rewards = []

print("Starting training...")
for i_episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for t in count():
        # Agent selects an action
        action = select_action(state)
        
        # Environment processes the action
        next_state, reward, done = env.step(action.item(), state)
        total_reward += reward.item()
        
        # Store memory in replay buffer
        memory.push(state, action, next_state, reward)
        
        # Move to next state
        state = next_state
        
        # Perform one step of the optimization
        optimize_model()
        
        # Update target network parameters
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            episode_rewards.append(total_reward)
            if (i_episode + 1) % 10 == 0:
                print(f"Episode {i_episode + 1}/{num_episodes} completed. Total Reward: {total_reward:.2f}")
            break

print('Training Complete.')

# GENERATE CONVERGENCE PLOT
plt.figure(figsize=(10, 5))
plt.title('Phase 1 Convergence: Reward vs. Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward per Episode (closer to 0 is better)')

# Plot raw rewards
plt.plot(episode_rewards, label='Episode Reward', color='lightgray')

# Plot moving average to see the trend clearly
rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float)
if len(rewards_tensor) >= 20:
    means = rewards_tensor.unfold(0, 20, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(19), means)) # Pad the start
    plt.plot(means.numpy(), label='20-Episode Moving Average', color='blue')

plt.legend()
plt.grid(True)
plt.savefig('phase1_convergence.png')
print('Chart saved as phase1_convergence.png')