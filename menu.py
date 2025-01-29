import pygame
import sys
from gui import CheckersGUI
from DQNAgent import DQNAgent
import torch
import math
import os

class Menu:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.WIDTH = 800  # Increased width for more space
        self.HEIGHT = 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Checkers AI')
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GOLD = (255, 215, 0)
        self.DARK_RED = (139, 0, 0)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 90)
        self.menu_font = pygame.font.Font(None, 50)
        
        # Menu items with descriptions
        self.menu_items = [
            {'text': 'Play vs AI', 'desc': 'Challenge our trained AI opponent'},
            {'text': 'AI Demo', 'desc': 'Watch AI play against itself'},
            {'text': 'Training Mode', 'desc': 'See how the AI learns'},
            {'text': 'Settings', 'desc': 'Adjust game parameters'},
            {'text': 'Quit', 'desc': 'Exit game'}
        ]
        self.selected_item = 0
        
        # Animation variables
        self.time = 0
        self.hover_offset = 0
        
        # Create checkerboard background
        self.background = self.create_background()

        # Sounds
        self.sounds = {
            'menu_move': pygame.mixer.Sound('sounds/tick.wav'),
            'menu_select': pygame.mixer.Sound('sounds/click.wav'),
        }

    def create_background(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        size = 50
        for y in range(0, self.HEIGHT, size):
            for x in range(0, self.WIDTH, size):
                if (x + y) // size % 2 == 0:
                    pygame.draw.rect(surface, (40, 40, 40), (x, y, size, size))
                else:
                    pygame.draw.rect(surface, (20, 20, 20), (x, y, size, size))
        return surface

    def draw_menu(self):
        # Draw animated background
        self.screen.blit(self.background, (0, 0))
        
        # Update animation time
        self.time += 0.05
        self.hover_offset = math.sin(self.time) * 5
        
        # Draw title with glow effect
        title_shadow = self.title_font.render('Checkers AI', True, self.DARK_RED)
        title = self.title_font.render('Checkers AI', True, self.GOLD)
        shadow_pos = (self.WIDTH/2 - 2, 102)
        title_pos = (self.WIDTH/2, 100)
        
        for pos in [(shadow_pos), (title_pos)]:
            text = title_shadow if pos == shadow_pos else title
            text_rect = text.get_rect(center=pos)
            self.screen.blit(text, text_rect)
        
        # Draw menu items with hover effect and descriptions
        for i, item in enumerate(self.menu_items):
            # Calculate position with hover effect for selected item
            base_y = 250 + i * 70
            y_pos = base_y + (self.hover_offset if i == self.selected_item else 0)
            
            # Draw selection indicator
            if i == self.selected_item:
                indicator_points = [
                    (self.WIDTH/2 - 140, y_pos),
                    (self.WIDTH/2 - 120, y_pos - 10),
                    (self.WIDTH/2 - 120, y_pos + 10)
                ]
                pygame.draw.polygon(self.screen, self.RED, indicator_points)
            
            # Draw menu item
            color = self.GOLD if i == self.selected_item else self.WHITE
            text = self.menu_font.render(item['text'], True, color)
            rect = text.get_rect(center=(self.WIDTH/2, y_pos))
            self.screen.blit(text, rect)
            
            # Draw description for selected item
            if i == self.selected_item:
                desc_font = pygame.font.Font(None, 30)
                desc = desc_font.render(item['desc'], True, (150, 150, 150))
                desc_rect = desc.get_rect(center=(self.WIDTH/2, y_pos + 30))
                self.screen.blit(desc, desc_rect)
        
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.sounds['menu_move'].play()  # Play sound on up movement
                        self.selected_item = (self.selected_item - 1) % len(self.menu_items)
                    elif event.key == pygame.K_DOWN:
                        self.sounds['menu_move'].play()  # Play sound on down movement
                        self.selected_item = (self.selected_item + 1) % len(self.menu_items)
                    elif event.key == pygame.K_RETURN:
                        if self.menu_items[self.selected_item]['text'] == 'Play vs AI':
                            self.sounds['menu_move'].play()
                            self.start_game()
                        elif self.menu_items[self.selected_item]['text'] == 'AI Demo':
                            self.sounds['menu_move'].play()
                            self.start_ai_demo()
                        elif self.menu_items[self.selected_item]['text'] == 'Quit':
                            self.sounds['menu_move'].play()
                            running = False
            
            self.draw_menu()
            clock.tick(60)

    def start_game(self):
        try:
            # Initialize agent with correct parameters
            agent = DQNAgent(state_size=36, action_size=1296)
            
            # Try different model paths
            model_paths = [
                'checkpoints/final_model.pt',
                'checkpoints/best_model.pth',
                'checkpoints/model_final_e100.pth'
            ]
            
            model_loaded = False
            for path in model_paths:
                try:
                    if os.path.exists(path):
                        print(f"Loading model from {path}")
                        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
                        agent.epsilon = 0  # No exploration in play mode
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not model_loaded:
                print("No trained model found. Using untrained agent.")
            
            game = CheckersGUI()
            game.player_turn = -1  # Human starts as white (-1), AI is red (1)
            game.run(agent)
            
        except Exception as e:
            print(f"Game error: {e}")
        finally:
            # Properly reinitialize pygame
            pygame.quit()
            pygame.init()
            pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.screen = pygame.display.get_surface()

    def start_ai_demo(self):
        """Start AI vs AI demo game"""
        try:
            # Initialize two agents with different exploration strategies
            agent1 = DQNAgent(state_size=36, action_size=1296)
            agent2 = DQNAgent(state_size=36, action_size=1296)
            
            # Load models for both agents
            model_paths = [
                'checkpoints/final_model.pt',
                'checkpoints/best_model.pth',
                'checkpoints/model_final_e100.pth'
            ]
            
            # Load models for both agents
            for agent in [agent1, agent2]:
                model_loaded = False
                for path in model_paths:
                    try:
                        if os.path.exists(path):
                            print(f"Loading model from {path}")
                            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
                            agent.epsilon = 0  # No exploration in demo mode
                            model_loaded = True
                            break
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        continue
                
                if not model_loaded:
                    print("No trained model found. Using untrained agent.")
            
            game = CheckersGUI()
            game.player_turn = 1  # Red starts
            game.is_demo = True  # Add this flag to GUI
            game.run_demo(agent1, agent2)  # New method for AI demo
            
        except Exception as e:
            print(f"Demo error: {e}")
        finally:
            # Properly reinitialize pygame
            pygame.quit()
            pygame.init()
            pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.screen = pygame.display.get_surface()

if __name__ == "__main__":
    menu = Menu()
    menu.run() 