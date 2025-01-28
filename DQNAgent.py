import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=36, action_size=36*36, hidden_size=128):
        self.state_size = state_size  # 6x6 board flattened
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network and target network
        self.q_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        # Convert state to numpy array if it isn't already
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, valid_moves):
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
            
        # Convert state to tensor efficiently
        state = torch.FloatTensor(np.asarray(state, dtype=np.float32).flatten()).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state)
        
        # Filter only valid moves
        valid_q_values = [action_values[self.encode_action(move)] for move in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        return valid_moves[best_move_idx]
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert batch of states to tensor efficiently
        states = np.vstack([s.flatten() for s, _, _, _, _ in batch])
        next_states = np.vstack([ns.flatten() for _, _, _, ns, _ in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        target = current_q_values.clone()
        
        for i, (_, action, reward, _, done) in enumerate(batch):
            if done:
                target[i][self.encode_action(action)] = reward
            else:
                target[i][self.encode_action(action)] = reward + self.gamma * torch.max(next_q_values[i])
        
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target)
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos 