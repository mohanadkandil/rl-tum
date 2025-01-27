import pygame
import numpy as np
from checkers_env import checkers_env

# Initialize Pygame
pygame.init()

# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class CheckersGUI:
    def __init__(self, board_size=6):
        self.board_size = board_size
        self.window_size = 800
        self.square_size = self.window_size // board_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Checkers')
        self.env = checkers_env(board_size=board_size)
        self.selected_piece = None
        self.valid_moves = []
        self.player_turn = 1
        self.game_over = False
        self.winner = 0
        self.font = pygame.font.Font(None, 74)  # For game over text

    def draw_board(self):
        self.screen.fill(BLACK)
        for row in range(self.board_size):
            for col in range(row % 2, self.board_size, 2):
                pygame.draw.rect(self.screen, GREY, 
                               (col * self.square_size, row * self.square_size, 
                                self.square_size, self.square_size))

    def draw_pieces(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.env.board[row][col]
                if piece != 0:
                    # Calculate center of square
                    x = col * self.square_size + self.square_size // 2
                    y = row * self.square_size + self.square_size // 2
                    
                    # Draw piece
                    if abs(piece) == 1:
                        color = RED if piece > 0 else WHITE
                        pygame.draw.circle(self.screen, color, (x, y), 
                                         self.square_size // 2 - 10)
                    else:  # King piece
                        color = RED if piece > 0 else WHITE
                        pygame.draw.circle(self.screen, color, (x, y), 
                                         self.square_size // 2 - 10)
                        pygame.draw.circle(self.screen, BLUE, (x, y), 
                                         self.square_size // 4)

    def draw_valid_moves(self):
        for move in self.valid_moves:
            row, col = move[2], move[3]  # End position of move
            x = col * self.square_size + self.square_size // 2
            y = row * self.square_size + self.square_size // 2
            pygame.draw.circle(self.screen, GREEN, (x, y), 15)

    def draw_selected(self):
        if self.selected_piece:
            row, col = self.selected_piece
            x = col * self.square_size + self.square_size // 2
            y = row * self.square_size + self.square_size // 2
            pygame.draw.circle(self.screen, GREEN, (x, y), 
                             self.square_size // 2 - 10, 4)

    def draw_game_over(self):
        """Draw game over screen"""
        if self.game_over:
            # Create semi-transparent overlay
            overlay = pygame.Surface((self.window_size, self.window_size))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(128)
            self.screen.blit(overlay, (0, 0))
            
            # Create game over text
            winner_text = "Red Wins!" if self.winner == 1 else "White Wins!"
            text = self.font.render(winner_text, True, (255, 215, 0))  # Gold color
            
            # Center the text
            text_rect = text.get_rect(center=(self.window_size/2, self.window_size/2))
            self.screen.blit(text, text_rect)
            
            # Add restart instruction
            restart_font = pygame.font.Font(None, 36)
            restart_text = restart_font.render("Press R to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.window_size/2, self.window_size/2 + 50))
            self.screen.blit(restart_text, restart_rect)

    def update_display(self):
        self.draw_board()
        self.draw_pieces()
        self.draw_valid_moves()
        self.draw_selected()
        self.draw_game_over()  # Add game over overlay if game is over
        pygame.display.flip()

    def handle_click(self, pos):
        if self.game_over:
            return

        row = pos[1] // self.square_size
        col = pos[0] // self.square_size
        piece = self.env.board[row][col]

        if self.selected_piece:
            move = self.try_move(row, col)
            if move:
                self.make_move(move)
            elif piece * self.player_turn > 0:
                self.select_piece(row, col)
        elif piece * self.player_turn > 0:
            self.select_piece(row, col)

    def select_piece(self, row, col):
        self.selected_piece = (row, col)
        self.valid_moves = [move for move in self.env.valid_moves(self.player_turn)
                          if move[0] == row and move[1] == col]

    def try_move(self, row, col):
        for move in self.valid_moves:
            if move[2] == row and move[3] == col:
                return move
        return None

    def make_move(self, move):
        try:
            board, reward, additional_moves = self.env.step(move, self.player_turn)
            
            if not additional_moves:
                self.player_turn = -self.player_turn
                winner = self.env.game_winner(board)
                if winner != 0:
                    self.game_over = True
                    self.winner = winner
                    print(f"Player {winner} wins!")
            
            self.selected_piece = None
            self.valid_moves = []
            
        except ValueError as e:
            print(f"Invalid move: {e}")

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    pos = pygame.mouse.get_pos()
                    self.handle_click(pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.__init__(self.board_size)
                    elif event.key == pygame.K_q:  # Quit game
                        running = False

            self.update_display()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    game = CheckersGUI()
    game.run()