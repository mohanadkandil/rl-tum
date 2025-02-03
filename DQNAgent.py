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
        self.unique_states = set()  # Track unique states
        self.priority_epsilon = 1e-6  # Add this line - small constant to prevent zero priorities
        
    def __len__(self):  # Add this method
        return len(self.memory)
        
    def push(self, state, action, reward, next_state, done):
        # Convert state to hashable format
        state_hash = hash(state.tobytes())
        
        # Only add if state is unique or randomly replace
        if state_hash not in self.unique_states or random.random() < 0.1:
            max_priority = np.max(self.priorities) if self.memory else 1.0
            
            if len(self.memory) < self.capacity:
                self.memory.append((state, action, reward, next_state, done))
            else:
                # Remove old state hash if replacing
                old_state = self.memory[self.position][0]
                self.unique_states.remove(hash(old_state.tobytes()))
                self.memory[self.position] = (state, action, reward, next_state, done)
            
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
            self.unique_states.add(state_hash)
        
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
            self.priorities[idx] = abs(error) + self.priority_epsilon  # Now this will work

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
        self.memory = PriorityReplayBuffer(100000, alpha=0.6)  # Initialize with alpha
        
        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.997  # Exploration decay rate
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.priority_beta = 0.4  # Add this line - importance sampling parameter
        
        # H100 specific optimizations
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
            
            # Enable TF32 and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True  # For reproducibility
            
            # Set higher memory fraction for GPU
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
            
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
        else:
            self.device = torch.device("cpu")
            print("WARNING: No GPU found, using CPU")
        
        def create_network():
            model = nn.Sequential(
                nn.Linear(state_size, 256),  # Reduced from 1024
                nn.ReLU(),
                nn.Dropout(0.3),  # Increased dropout
                nn.Linear(256, 256),  # Reduced from 1024
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, action_size)
            )
            
            # Initialize weights with smaller values
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0)
            
            return model.to(self.device)
        
        # Create networks and ensure they're on GPU
        self.q_network = create_network()
        self.target_network = create_network()
        
        # Use mixed precision training only if CUDA is available
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Move optimizer to GPU
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        
        # Add priority replay parameters
        self.priority_alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
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
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        batch, indices, weights = self.memory.sample(self.batch_size, self.priority_beta)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute loss
        self.q_network.train()
        loss = self._compute_loss(batch)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()
        
        # Update priorities in replay buffer
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