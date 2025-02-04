import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.multiprocessing as mp
import torch.nn.functional as F

mp.set_start_method('spawn', force=True)  # Add this at the top of the file

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

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def __len__(self):
        return len(self.memory)
        
    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return []
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        return samples, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Add small constant to prevent zero priority

class DQNAgent:
    def __init__(self, state_size=36, action_size=1296):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = 100000
        self.memory = PriorityReplayBuffer(self.memory_size, alpha=0.6)
        
        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.997  # Exploration decay rate
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.priority_beta = 0.4
        
        # Device configuration
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("WARNING: No GPU found, using CPU")
        
        # Create networks
        self.q_network = self.create_network()
        self.target_network = self.create_network()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
    def create_network(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        
        # Initialize weights properly
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        return model.to(self.device)
        
    def remember(self, state, action, reward, next_state, done):
        # Calculate priority based on reward magnitude
        priority = abs(reward) + 1.0  # Add 1 to ensure non-zero priority
        
        # Store experience with priority
        self.memory.push(state, action, reward, next_state, done)
        
        # Keep memory size in check
        if len(self.memory) > self.memory_size:
            # Remove lowest priority experience
            min_priority_idx = min(range(len(self.memory)), 
                                 key=lambda i: self.memory[i][5])
            self.memory.memory.pop(min_priority_idx)
        
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
            
        # Get batch using priority sampling
        batch, indices = self.memory.sample(self.batch_size)
        
        # Compute loss
        self.q_network.train()
        loss = self._compute_loss(batch)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = self._compute_td_errors(batch)
        self.memory.update_priorities(indices, td_errors.cpu().numpy())

    def _compute_td_errors(self, batch):
        """Compute TD errors for prioritized replay"""
        states = torch.FloatTensor(np.vstack([s.flatten() for s, _, _, _, _ in batch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([ns.flatten() for _, _, _, ns, _ in batch])).to(self.device)
        actions = torch.LongTensor([[self.encode_action(a)] for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        return torch.abs(target_q_values - current_q_values)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos 

    def _compute_loss(self, batch):
        """Compute loss for a batch of transitions"""
        states = torch.FloatTensor(np.vstack([s.flatten() for s, _, _, _, _ in batch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([ns.flatten() for _, _, _, ns, _ in batch])).to(self.device)
        actions = torch.LongTensor([[self.encode_action(a)] for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss with Huber loss (more stable than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        return loss 

    def predict_piece_count(self, state):
        """Auxiliary task to predict piece count"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).to(self.device)
            # Use first layer of network for prediction
            piece_count_pred = self.q_network[0](state_tensor).mean().item()
        return piece_count_pred 