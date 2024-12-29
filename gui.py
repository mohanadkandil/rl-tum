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
CROWN = pygame.transform.scale(pygame.image.load('crown.png'), (44, 25))

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')

def draw_board(screen, board):
    screen.fill(BLACK)
    for row in range(ROWS):
        for col in range(row % 2, COLS, 2):
            pygame.draw.rect(screen, GREY, (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != 0:
                if piece == 1:
                    color = RED
                elif piece == -1:
                    color = WHITE
                elif piece == 2:
                    color = RED
                elif piece == -2:
                    color = WHITE
                pygame.draw.circle(screen, color, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 10)
                if piece == 2 or piece == -2:
                    screen.blit(CROWN, (col * SQUARE_SIZE + SQUARE_SIZE // 2 - CROWN.get_width() // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2 - CROWN.get_height() // 2))

def main():
    env = checkers_env()
    board = env.board
    clock = pygame.time.Clock()
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_board(screen, board)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()