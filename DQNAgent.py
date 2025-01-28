import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.multiprocessing as mp
import torch.nn.functional as F

class ParallelDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ParallelDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout to prevent overfitting
            nn.Linear(hidden_size, hidden_size * 2),  # Wider network
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),  # Bottleneck
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Enable parallel processing
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.network = nn.DataParallel(self.network)

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=36, action_size=1296):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.995  
        self.learning_rate = 0.0001
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def create_network():
            return nn.Sequential(
                nn.Linear(state_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, action_size)
            ).to(self.device)
        
        self.q_network = create_network()
        self.target_network = create_network()
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        # Store as numpy arrays
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, valid_moves):
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        # Set to eval mode for prediction
        self.q_network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32).flatten()).unsqueeze(0).to(self.device)
            action_values = self.q_network(state).cpu()
        
        # Filter only valid moves
        valid_q_values = [action_values[0][self.encode_action(move)] for move in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        return valid_moves[best_move_idx]
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.vstack([s.flatten() for s, _, _, _, _ in batch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([ns.flatten() for _, _, _, ns, _ in batch])).to(self.device)
        
        # Set networks to appropriate modes
        self.q_network.train()  # Enable dropout during training
        self.target_network.eval()  # Disable dropout for target network
        
        current_q_values = self.q_network(states)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
        
        target = current_q_values.clone()
        
        for i, (_, action, reward, _, done) in enumerate(batch):
            if done:
                target[i][self.encode_action(action)] = reward
            else:
                target[i][self.encode_action(action)] = reward + self.gamma * torch.max(next_q_values[i])
        
        loss = F.mse_loss(current_q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos 