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

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.priority_epsilon = 1e-6  # Add this here
        
    def __len__(self):  # Add this method
        return len(self.memory)
        
    def push(self, state, action, reward, next_state, done):
        # New experiences get max priority
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta):
        if len(self.memory) == 0:
            return [], [], []
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.memory[idx] for idx in indices]
        return samples, indices, weights
        
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.priority_epsilon

class DQNAgent:
    def __init__(self, state_size=36, action_size=1296):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PriorityReplayBuffer(100000, 0.6)
        
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
        
        # Add priority replay parameters
        self.priority_alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.priority_beta = 0.4   # Importance sampling correction (starts low, annealed to 1)
        self.priority_epsilon = 1e-6  # Small constant to prevent zero priorities
        
    def remember(self, state, action, reward, next_state, done):
        # Store as numpy arrays
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.push(state, action, reward, next_state, done)
        
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
        if len(self.memory.memory) < self.batch_size:
            return
            
        # Sample with priorities
        batch, indices, weights = self.memory.sample(self.batch_size, self.priority_beta)
        weights = torch.FloatTensor(weights).to(self.device)
        
        states = torch.FloatTensor(np.vstack([s.flatten() for s, _, _, _, _ in batch])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([ns.flatten() for _, _, _, ns, _ in batch])).to(self.device)
        
        self.q_network.train()
        self.target_network.eval()
        
        current_q_values = self.q_network(states)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
        
        target = current_q_values.clone()
        td_errors = []
        
        for i, (_, action, reward, _, done) in enumerate(batch):
            if done:
                target_value = reward
            else:
                target_value = reward + self.gamma * torch.max(next_q_values[i])
            
            # Calculate TD error for priority update
            current_value = current_q_values[i][self.encode_action(action)]
            td_error = abs(target_value - current_value.item())
            td_errors.append(td_error)
            
            target[i][self.encode_action(action)] = target_value
        
        # Calculate element-wise loss
        losses = F.mse_loss(current_q_values, target, reduction='none')
        
        # Apply weights to batch dimension only
        weighted_loss = (weights.unsqueeze(1) * losses.mean(dim=1)).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos 