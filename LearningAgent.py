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
        :param step_size:
        :param epsilon:
        :param env:
        '''

        self.step_size = step_size  # Learning rate
        self.epsilon = epsilon      # Exploration rate
        self.env = env
        
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
        if np.random.random() < self.epsilon:
            valid_moves = self.env.valid_moves(self.env.player)
            if not valid_moves:
                return None
            return valid_moves[np.random.randint(len(valid_moves))]
        
        valid_moves = self.env.valid_moves(self.env.player)
        if not valid_moves:
            return None
            
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
        
        # Q-learning update formula
        new_q = current_q + self.step_size * (reward + GAMMA * max_next_q - current_q)
        
        # Update Q-table
        state_key = self.get_state_key(state)
        self.q_table[state_key][str(action)] = new_q

    def evaluation(self, board):
        '''
        Evaluate the current board state for international checkers (6x6 board).
        Returns a score indicating how favorable the position is.
        
        Key differences from American checkers:
        1. Pieces can move backwards even when not kings
        2. Kings have longer range movement
        3. Forced capture rules are stricter
        4. Different initial setup
        '''
        
        # Count regular pieces and kings for both players
        player_pieces = 0
        opponent_pieces = 0
        player_kings = 0
        opponent_kings = 0
        
        # Mobility scores (count possible moves)
        player_mobility = len(self.env.valid_moves(self.env.player))
        opponent_mobility = len(self.env.valid_moves(-self.env.player))
        
        # Control of key squares (diagonals and center)
        key_squares = [
            (1,1), (1,3), (1,5),  
            (2,2), (2,3), (3,2), (3,3),  
            (4,1), (4,3), (4,5)  
        ]
        player_key_squares = 0
        opponent_key_squares = 0
        
        # Evaluate each position on the board
        for row in range(6):
            for col in range(6):
                piece = board[row][col]
                
                if piece != 0:  # If square is not empty
                    # Count pieces and kings
                    if piece == self.env.player:
                        if abs(piece) == 1:
                            player_pieces += 1
                        else:
                            player_kings += 1
                            
                    elif piece == -self.env.player:
                        if abs(piece) == 1:
                            opponent_pieces += 1
                        else:
                            opponent_kings += 1
                    
                    # Count control of key squares
                    if (row, col) in key_squares:
                        if piece == self.env.player:
                            player_key_squares += 1
                        elif piece == -self.env.player:
                            opponent_key_squares += 1
        
        # Calculate final score with weights adjusted for international rules
        piece_score = (player_pieces - opponent_pieces) * 100
        king_score = (player_kings - opponent_kings) * 300  
        mobility_score = (player_mobility - opponent_mobility) * 30  
        position_score = (player_key_squares - opponent_key_squares) * 50
        
        king_bonus = 0
        if player_kings > opponent_kings:
            king_bonus = 100
        
        total_score = piece_score + king_score + mobility_score + position_score + king_bonus
        
        return total_score