import checkers_env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        '''
        :param step_size: Learning rate
        :param epsilon: Exploration rate
        :param env: Checkers environment
        '''
        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        self.gamma = 0.99  # Add discount factor here
        self.q_table = {}

    def get_state_key(self, state):
        """Convert board state to a string key for Q-table"""
        return str(state.flatten().tolist())
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if str(action) not in self.q_table[state_key]:
            self.q_table[state_key][str(action)] = 0.0
        return self.q_table[state_key][str(action)]
    
    def select_action(self):
        """Select action using epsilon-greedy strategy"""
        # Get valid moves for the current player
        valid_moves = self.env.valid_moves(self.env.player)
        
        # If no valid moves available
        if not valid_moves:
            return None
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            # Explore: random action from valid moves
            return valid_moves[np.random.randint(len(valid_moves))]
        
        # Exploit: choose best known action from valid moves
        q_values = [self.get_q_value(self.env.board, action) for action in valid_moves]
        max_q_index = np.argmax(q_values)
        return valid_moves[max_q_index]
    
    def learning(self, state, action, reward, next_state):
        """Q-learning update"""
        if action is None:
            return
            
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        next_valid_moves = self.env.valid_moves(self.env.player)
        if next_valid_moves:
            next_q_values = [self.get_q_value(next_state, next_action) 
                           for next_action in next_valid_moves]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0
        
        # Q-learning update formula using self.gamma instead of GAMMA
        new_q = current_q + self.step_size * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        state_key = self.get_state_key(state)
        self.q_table[state_key][str(action)] = new_q

    def evaluation(self, board):
        """
        Evaluate board position for American checkers
        Returns a score indicating how favorable the position is
        """
        # Count pieces and kings
        player_pieces = np.sum(board == self.env.player)
        player_kings = np.sum(board == self.env.player * 2)
        opponent_pieces = np.sum(board == -self.env.player)
        opponent_kings = np.sum(board == -self.env.player * 2)
        
        # Count mobility (available moves)
        player_mobility = len(self.env.valid_moves(self.env.player))
        opponent_mobility = len(self.env.valid_moves(-self.env.player))
        
        # Control of center squares (more valuable)
        center_squares = [(3,3), (3,4), (4,3), (4,4)]
        player_center = sum(1 for r, c in center_squares 
                           if board[r][c] in [self.env.player, self.env.player * 2])
        opponent_center = sum(1 for r, c in center_squares 
                             if board[r][c] in [-self.env.player, -self.env.player * 2])
        
        # Calculate weighted score
        piece_score = (player_pieces - opponent_pieces) * 100
        king_score = (player_kings - opponent_kings) * 200
        mobility_score = (player_mobility - opponent_mobility) * 30
        center_score = (player_center - opponent_center) * 50
        
        total_score = piece_score + king_score + mobility_score + center_score
        
        return total_score