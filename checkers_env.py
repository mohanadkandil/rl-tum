import numpy as np

# Helper methods
def in_board(row, col):
    return 0 <= row < 6 and 0 <= col < 6

def board_pos_empty(board, row, col):
    return board[row, col] == 0


class checkers_env:

    def __init__(self, board_size=6):  # Default to 6x6
        self.board_size = board_size
        self.board = self.initialize_board()
        self.player = 1  # 1 for red (top), -1 for white (bottom)
        self.selected_piece = None


    def initialize_board(self):
        """Initialize board for checkers"""
        board = np.zeros((self.board_size, self.board_size))
        
        # Set up red pieces (player 1)
        for row in range(2):  # Only 2 rows for 6x6
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    board[row][col] = 1
                    
        # Set up white pieces (player -1)
        for row in range(self.board_size - 2, self.board_size):  # Only 2 rows for 6x6
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    board[row][col] = -1
                    
        return board


    def reset(self):
        """Reset the game to initial state"""
        self.board = self.initialize_board()
        self.player = 1
        return self.board


    def valid_moves(self, player):
        """Get all valid moves for current player"""
        moves = []
        capture_moves = []  # Separate list for capture moves

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row][col]
                if piece == player or piece == player * 2:  # Regular piece or king
                    # Check all possible moves for this piece
                    piece_moves = self._get_piece_moves(row, col, player)
                    piece_captures = self._get_piece_captures(row, col, player)
                    
                    moves.extend(piece_moves)
                    capture_moves.extend(piece_captures)
        
        # If captures available, only return captures (forced capture rule)
        return capture_moves if capture_moves else moves

    def _get_piece_moves(self, row, col, player):
        """Get regular moves for a piece"""
        moves = []
        piece = self.board[row][col]
        directions = []
        
        # Regular pieces can only move forward
        if piece == 1:  # Red moves down
            directions = [(1, -1), (1, 1)]
        elif piece == -1:  # White moves up
            directions = [(-1, -1), (-1, 1)]
        # Kings can move in all directions
        elif abs(piece) == 2:
            directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_pos(new_row, new_col) and self.board[new_row][new_col] == 0:
                moves.append([row, col, new_row, new_col])
                
        return moves

    def _get_piece_captures(self, row, col, player):
        """Get capture moves for a piece"""
        captures = []
        piece = self.board[row][col]
        directions = []
        
        if piece == 1 or abs(piece) == 2:
            directions.extend([(1, -1), (1, 1)])
        if piece == -1 or abs(piece) == 2:
            directions.extend([(-1, -1), (-1, 1)])

        for dr, dc in directions:
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if self._is_valid_pos(jump_row, jump_col):
                middle_row, middle_col = row + dr, col + dc
                if (self.board[middle_row][middle_col] == -player or 
                    self.board[middle_row][middle_col] == -player * 2):
                    if self.board[jump_row][jump_col] == 0:
                        captures.append([row, col, jump_row, jump_col])
        
        return captures

    def _is_valid_pos(self, row, col):
        """Check if position is within board"""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def step(self, action, player):
        """Execute move and return new state, reward, additional_moves, and done"""
        try:
            start_row, start_col, end_row, end_col = action
            piece = self.board[start_row][start_col]
            reward = 0
            done = False
            additional_moves = []
            
            # Make the move
            self.board[start_row][start_col] = 0
            self.board[end_row][end_col] = piece
            
            # Capture reward - keep high to encourage aggressive play
            if abs(start_row - end_row) == 2:
                mid_row = (start_row + end_row) // 2
                mid_col = (start_col + end_col) // 2
                captured_piece = self.board[mid_row][mid_col]
                self.board[mid_row][mid_col] = 0
                reward += 15 if abs(captured_piece) == 2 else 10  # High reward for captures
                
                # Check for additional captures
                additional_moves = [move for move in self.valid_moves(player) 
                                  if move[0] == end_row and move[1] == end_col and abs(move[2] - move[0]) == 2]
            
            # Strategic position rewards - encourage forward movement
            if player == 1:  # Red moving down
                reward += 1.0 * (end_row - start_row)  # Significant reward for forward progress
                if end_row == self.board_size-1:  # King promotion
                    reward += 20  # Big reward for getting a king
            else:  # White moving up
                reward += 1.0 * (start_row - end_row)
                if end_row == 0:  # King promotion
                    reward += 20
            
            # Center control reward - important for strategy
            center_squares = [(2,2), (2,3), (3,2), (3,3)]
            if (end_row, end_col) in center_squares:
                reward += 3  # Good reward for controlling center
            
            # King mobility reward - encourage active kings
            if abs(piece) == 2:
                valid_moves = self._get_piece_moves(end_row, end_col, player)
                reward += 0.5 * len(valid_moves)  # Reward mobile kings
            
            # Win/Loss rewards - keep high stakes
            winner = self.game_winner(self.board)
            if winner == player:
                reward += 50  # Big reward for winning
                done = True
            elif winner == -player:
                reward -= 50  # Big penalty for losing
                done = True
            
            # Defensive bonus - reward protecting back row
            if player == 1 and start_row == 0 and piece > 0:  # Red defending back
                reward += 2
            elif player == -1 and start_row == self.board_size-1 and piece < 0:  # White defending back
                reward += 2
            
            return self.board, reward, additional_moves, done
            
        except ValueError as e:
            raise ValueError("Invalid move")

    def game_winner(self, board):
        """Check if game is won"""
        red_pieces = np.sum(board > 0)
        white_pieces = np.sum(board < 0)
        
        if white_pieces == 0:
            return 1
        elif red_pieces == 0:
            return -1
        elif len(self.valid_moves(1)) == 0:
            return -1
        elif len(self.valid_moves(-1)) == 0:
            return 1
        return 0

    def render(self):
        """Print current board state"""
        print(f"\n  {''.join([str(i) + ' ' for i in range(self.board_size)])}")
        for row in range(self.board_size):
            print(f"{row}", end=" ")
            for col in range(self.board_size):
                piece = self.board[row][col]
                if piece == 0:
                    print(".", end=" ")
                elif piece == 1:
                    print("r", end=" ")
                elif piece == -1:
                    print("w", end=" ")
                elif piece == 2:
                    print("R", end=" ")
                elif piece == -2:
                    print("W", end=" ")
            print()
        print()