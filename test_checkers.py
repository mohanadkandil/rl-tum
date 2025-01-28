import unittest
import numpy as np
from checkers_env import checkers_env
from LearningAgent import LearningAgent

class TestCheckersLogic(unittest.TestCase):
    def setUp(self):
        """Set up a fresh board before each test"""
        self.env = checkers_env()
        self.agent = LearningAgent(step_size=0.1, epsilon=0.1, env=self.env)

    def test_initial_board(self):
        """Test if board is initialized correctly"""
        expected_board = np.array([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1]
        ])
        np.testing.assert_array_equal(self.env.board, expected_board)

    def test_valid_moves(self):
        """Test if valid moves are generated correctly"""
        moves = self.env.valid_moves(1)
        self.assertEqual(len(moves), 5)
        
        expected_moves = [
            [1, 1, 2, 0],
            [1, 1, 2, 2],
            [1, 3, 2, 2],
            [1, 3, 2, 4],
            [1, 5, 2, 4]
        ]
        self.assertEqual(sorted(moves), sorted(expected_moves))

    def test_capture_move(self):
        """Test if piece capture works correctly"""
        self.env.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        action = [1, 1, 3, 3]  
        self.env.step(action, 1)
        
        self.assertEqual(self.env.board[2, 2], 0)
        self.assertEqual(self.env.board[3, 3], 1)
        self.assertEqual(self.env.board[1, 1], 0)

    def test_king_promotion(self):
        """Test if pieces are promoted to kings correctly"""
        self.env.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        action = [4, 2, 5, 3]
        self.env.step(action, 1)
        
        self.assertEqual(self.env.board[5, 3], 2)  # 2 represents a king

    def test_double_jump(self):
        """Test if double jumps are handled correctly"""
        self.env.board = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        action = [1, 1, 3, 3]
        board, reward, additional_moves = self.env.step(action, 1)
        
        self.assertTrue(len(additional_moves) > 0)
        self.assertEqual(self.env.board[2, 2], 0)

    def test_game_winner(self):
        """Test if game winner is determined correctly"""
        self.env.board = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        self.assertEqual(self.env.game_winner(self.env.board), 1)

if __name__ == '__main__':
    unittest.main() 