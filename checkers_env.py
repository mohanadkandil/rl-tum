
import numpy as np


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
        '''
        A possible format could be [start_row, start_col, end_row, end_col], there are normal moves and moves with capture. Pieces could be king or normal.
        '''


    def capture_piece(self, action):
        '''
        Assign 0 to the positions of captured pieces.
        '''
        

    
    def game_winner(self, board):

        '''
        return player 1 win or player -1 win or draw
        '''


    def step(self, action, player):
        '''
        The transition of board and incurred reward after player performs an action. Be careful about King
        '''

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