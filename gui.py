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
        self.winner = None
        self.font = pygame.font.Font(None, 74)
        
        # Add sounds
        self.sounds = {
            'piece_move': pygame.mixer.Sound('sounds/attack.wav'),
            'victory': pygame.mixer.Sound('sounds/victory.wav'),
        }
        self.victory_played = False
        
        # Add win state variables
        self.game_over = False
        self.winner = None
        
        # Add colors for win state
        self.WIN_COLOR = (0, 255, 0)  # Green
        self.LOSE_COLOR = (255, 0, 0)  # Red
        self.DRAW_COLOR = (255, 255, 0)  # Yellow

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
        """Draw game over message"""
        font = pygame.font.Font(None, 74)
        if self.winner == 1:
            text = "Red Wins!"
            color = self.WIN_COLOR if self.player_turn == 1 else self.LOSE_COLOR
        elif self.winner == -1:
            text = "White Wins!"
            color = self.WIN_COLOR if self.player_turn == -1 else self.LOSE_COLOR
        else:
            text = "Draw!"
            color = self.DRAW_COLOR
            
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
        
        # Add semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        # Draw text
        self.screen.blit(text_surface, text_rect)

    def update_display(self):
        # Draw board and pieces
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_pieces()
        
        # Draw win state if game is over
        if self.game_over:
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
        """Handle AI move and check for game over"""
        if self.player_turn == 1: 
            state = self.env.board.flatten()
            valid_moves = self.env.valid_moves(1)
            
            if valid_moves:
                action = agent.act(state, valid_moves)
                next_state, reward, done = self.env.step(action, 1)
                
                if done:
                    self.game_over = True
                    self.winner = self.env.game_winner(next_state)
                    return True
                    
                self.player_turn = -1 
                
            else:
                self.game_over = True
                self.winner = -1  
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

    def run_demo(self, agent1, agent2):
        """Run AI vs AI demo"""
        clock = pygame.time.Clock()
        running = True
        move_delay = 1000  # 1 second delay between moves
        last_move_time = pygame.time.get_ticks()
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.__init__()
                        self.is_demo = True
                        self.player_turn = 1
                    elif event.key == pygame.K_q:  # Quit demo
                        running = False
            
            # Make AI moves with delay
            if not self.game_over and current_time - last_move_time > move_delay:
                if self.player_turn == 1:  # Red's turn (agent1)
                    state = self.env.board.flatten()
                    valid_moves = self.env.valid_moves(1)
                    
                    if valid_moves:
                        action = agent1.act(state, valid_moves)
                        next_state, reward, done = self.env.step(action, 1)
                        
                        if done:
                            self.game_over = True
                            self.winner = self.env.game_winner(next_state)
                        else:
                            self.player_turn = -1  # Switch to white
                    else:
                        self.game_over = True
                        self.winner = -1
                    
                else:  # White's turn (agent2)
                    state = self.env.board.flatten()
                    valid_moves = self.env.valid_moves(-1)
                    
                    if valid_moves:
                        action = agent2.act(state, valid_moves)
                        next_state, reward, done = self.env.step(action, -1)
                        
                        if done:
                            self.game_over = True
                            self.winner = self.env.game_winner(next_state)
                        else:
                            self.player_turn = 1  # Switch to red
                    else:
                        self.game_over = True
                        self.winner = 1
                
                last_move_time = current_time
                
                # Play move sound
                if not self.game_over:
                    self.sounds['piece_move'].play()
            
            self.update_display()
            clock.tick(60)

if __name__ == "__main__":
    gui = CheckersGUI()
    gui.run()