
import numpy as np

# Helper methods
def in_board(row, col):
    return 0 <= row < 6 and 0 <= col < 6

def board_pos_empty(board, row, col):
    return board[row, col] == 0


class checkers_env:

    def __init__(self, board=None, player=None):

        self.board = self.initialize_board()
        self.player = 1


    def initialize_board(self):
        board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [-1, 0, -1, 0, -1, 0],
                      [0, -1, 0, -1, 0, -1]])
        return board


    def reset(self):
        self.board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [-1, 0, -1, 0, -1, 0],
                      [0, -1, 0, -1, 0, -1]])
        self.player = 1


    def valid_moves(self, player):
        """
        A possible format could be [start_row, start_col, end_row, end_col], there are normal moves and moves with capture. Pieces could be king or normal.
        """
        possible_moves = []

        for row in range(6):
            for col in range(6):
                piece = self.board[row, col]

                if piece == player or piece == 2 * player:
                    if piece == player:
                        if player == 1:
                            # Player is top side
                            directions = [(1, -1), (1, 1)]
                        else:
                            # Player is bottom side
                            directions = [(-1, -1), (-1, 1)]
                    else:
                        # Player is King
                        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

                    for drow, dcol in directions:
                        next_row = row + drow
                        next_col = col + dcol

                        # Check if next is within the board
                        if in_board(next_row, next_col):
                            target = self.board[next_row, next_col]

                            if target == 0:
                                # If target spot is empty
                                possible_moves.append([row, col, next_row, next_col])
                            else:
                                # Check if target is opponent
                                if (target != 0) and (np.sign(target) != np.sign(piece)):
                                    jump_row = next_row + drow
                                    jump_col = next_col + dcol

                                    if in_board(jump_row, jump_col) and board_pos_empty(self.board, jump_row, jump_col):
                                        possible_moves.append([row, col, jump_row, jump_col])
        return possible_moves

    def capture_piece(self, action):
        """
        Assign 0 to the positions of captured pieces.
        We define `action` as [start_row, start_col, end_row, end_col]
        """
        start_row, start_col, end_row, end_col = action

        # Calculate the position of the captured piece
        captured_row = (start_row + end_row) // 2
        captured_col = (start_col + end_col) // 2

        self.board[captured_row][captured_col] = 0

def game_winner(self, board):
    """
    return player 1 win or player -1 win or draw
    """
    if np.sum(board < 0) == 0:
        return 1
    elif np.sum(board > 0) == 0:
        return -1
    elif len(self.valid_moves(-1)) == 0:
        return -1
    elif len(self.valid_moves(1)) == 0:
        return 1
    else:
        return 0


def step(self, action, player):
    """
    The transition of board and incurred reward after player performs an action. Be careful about King
    """
    reward = 0 # change

    return [self.board, reward]


def render(self):
    for row in self.board:
        for square in row:
            if square == 1:
                piece = "|0"
            elif square == -1:
                piece = "|X"
            elif square == 2:
                piece = "|K"
            else:
                piece = "| "
            print(piece, end='')
        print("|")