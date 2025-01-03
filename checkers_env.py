
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
        Also check if the action is a capture move, return boolean
        """
        start_row, start_col, end_row, end_col = action

        # Calculate the position of the captured piece
        captured_row = (start_row + end_row) // 2
        captured_col = (start_col + end_col) // 2

        if abs(start_row - end_row) == 2 and self.board[captured_row][captured_col] != 0:
            self.board[captured_row][captured_col] = 0
            return True
        return False

    def step(self, action, player):
        start_row, start_col, end_row, end_col = action
        reward = 0

        if action in self.valid_moves(player):
            piece = self.board[start_row][start_col]
            self.board[start_row][start_col] = 0
            self.board[end_row][end_col] = piece

            # Check for King promotion
            if player == 1 and end_row == 5:
                self.board[end_row][end_col] = 2
            elif player == -1 and end_row == 0:
                self.board[end_row][end_col] = -2

            if self.capture_piece(action):
                reward = 1
                # Check for additional capture moves from the new position
                additional_moves = self.valid_moves(player)
                additional_moves = [move for move in additional_moves if
                                    move[0] == end_row and move[1] == end_col and abs(move[2] - move[0]) == 2]
                if additional_moves:
                    return [self.board, reward, additional_moves]

            if self.game_winner(self.board) == player:
                reward = 10
        else:
            raise ValueError("Invalid move")

        self.render()

        return [self.board, reward, []]

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

    def render(self):
        print("\n", end='')
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