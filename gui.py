import pygame
import numpy as np
from checkers_env import checkers_env

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 6, 6
SQUARE_SIZE = WIDTH // COLS

# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CROWN = pygame.transform.scale(pygame.image.load('crown.png'), (44, 25))

class CheckersGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Checkers')
        self.env = checkers_env()
        self.selected_piece = None
        self.valid_moves = []
        self.player_turn = -1  # Changed to -1 so human starts with white
        self.game_over = False
        self.winner = 0
        self.font = pygame.font.Font(None, 74)
        
        # Add sounds
        self.sounds = {
            'piece_move': pygame.mixer.Sound('sounds/attack.wav'),
            'victory': pygame.mixer.Sound('sounds/victory.wav'),
        }
        self.victory_played = False

    def draw_board(self):
        self.screen.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(self.screen, GREY, 
                               (col * SQUARE_SIZE, row * SQUARE_SIZE, 
                                SQUARE_SIZE, SQUARE_SIZE))
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.env.board[row][col]
                if piece != 0:
                    if piece == 1:
                        color = RED
                    elif piece == -1:
                        color = WHITE
                    elif piece == 2:
                        color = RED
                    elif piece == -2:
                        color = WHITE
                    pygame.draw.circle(self.screen, color, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 10)
                    if piece == 2 or piece == -2:
                        self.screen.blit(CROWN, (col * SQUARE_SIZE + SQUARE_SIZE // 2 - CROWN.get_width() // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2 - CROWN.get_height() // 2))
        for move in self.valid_moves:
            pygame.draw.circle(self.screen, GREEN, 
                             (move[3] * SQUARE_SIZE + SQUARE_SIZE // 2, 
                              move[2] * SQUARE_SIZE + SQUARE_SIZE // 2), 
                             8)

    def draw_pieces(self):
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.env.board[row][col]
                if piece != 0:
                    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    
                    if abs(piece) == 1:
                        color = RED if piece > 0 else WHITE
                        pygame.draw.circle(self.screen, color, (x, y), 
                                         SQUARE_SIZE // 2 - 10)
                    else:  # King piece
                        color = RED if piece > 0 else WHITE
                        pygame.draw.circle(self.screen, color, (x, y), 
                                         SQUARE_SIZE // 2 - 10)
                        pygame.draw.circle(self.screen, BLUE, (x, y), 
                                         SQUARE_SIZE // 4)

    def draw_valid_moves(self):
        for move in self.valid_moves:
            row, col = move[2], move[3]  # End position
            x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(self.screen, GREEN, (x, y), 8)  # Reduced from 15 to 8

    def draw_game_over(self):
        if self.game_over:
            if not self.victory_played:
                self.sounds['victory'].play()
                self.victory_played = True
            
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.fill(BLACK)
            overlay.set_alpha(128)
            self.screen.blit(overlay, (0, 0))
            
            winner_text = "Red Wins!" if self.winner == 1 else "White Wins!"
            text = self.font.render(winner_text, True, (255, 215, 0))
            text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2))
            self.screen.blit(text, text_rect)
            
            restart_font = pygame.font.Font(None, 36)
            restart_text = restart_font.render("Press R to Restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WIDTH/2, HEIGHT/2 + 50))
            self.screen.blit(restart_text, restart_rect)

    def update_display(self):
        self.draw_board()
        self.draw_pieces()
        self.draw_valid_moves()
        self.draw_game_over()
        pygame.display.flip()

    def handle_click(self, pos):
        if self.game_over:
            return

        row = pos[1] // SQUARE_SIZE
        col = pos[0] // SQUARE_SIZE
        piece = self.env.board[row][col]

        if self.selected_piece:
            move = self.try_move(row, col)
            if move:
                # Play sound when making a valid move
                self.sounds['piece_move'].play()
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
            
            self.selected_piece = None
            self.valid_moves = []
            
        except ValueError as e:
            print(f"Invalid move: {e}")

    def handle_ai_move(self, agent):
        """Handle AI move"""
        if self.player_turn == 1:  # AI plays as red (1)
            state = self.env.board.flatten()
            valid_moves = self.env.valid_moves(1)  # Get moves for red
            
            if valid_moves:
                action = agent.act(state, valid_moves)
                next_state, reward, done = self.env.step(action, 1)  # AI makes move as red
                
                if done:
                    print("Game over!")
                    return True
                    
                self.player_turn = -1  # Switch to human player (white)
                
            else:
                print("AI has no valid moves")
                return True
                
        return False

    def run(self, agent=None):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.player_turn == -1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.__init__()
                        self.player_turn = -1  # Reset to human's turn
                    elif event.key == pygame.K_q:  # Quit game
                        running = False

            # AI move if it's AI's turn
            if agent and self.player_turn == 1 and not self.game_over:  # AI plays as red (1)
                if self.handle_ai_move(agent):
                    continue

            self.update_display()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    gui = CheckersGUI()
    gui.run()