import pygame
import sys
from gui import CheckersGUI
from DQNAgent import DQNAgent
import torch
import math
from cloud_storage import S3Handler
import boto3
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

        self.s3 = S3Handler()
        self.available_models = self.s3.get_available_models()
        
        # Add model selection to menu
        self.menu_items.insert(1, {
            'text': 'Select Model', 
            'desc': 'Choose from available trained models'
        })

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
                        if self.menu_items[self.selected_item]['text'] == 'Select Model':
                            model_path = self.show_model_selection()
                            if model_path:
                                self.start_game(model_path)
                        elif self.menu_items[self.selected_item]['text'] == 'Play vs AI':
                            self.sounds['menu_move'].play()  # Play sound on selection
                            self.start_game()
                        elif self.menu_items[self.selected_item]['text'] == 'Quit':
                            self.sounds['menu_move'].play()  # Play sound before quitting
                            running = False
            
            self.draw_menu()
            clock.tick(60)

    def start_game(self, model_path=None):
        agent = DQNAgent(state_size=36, action_size=1296, hidden_size=256)
        if model_path:
            checkpoint = torch.load(model_path, weights_only=True)
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            agent.epsilon = 0
        
        game = CheckersGUI()
        game.player_turn = -1
        game.run(agent)
        
        # Reinitialize pygame display for menu after game ends
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def show_model_selection(self):
        s3_handler = S3Handler()
        models = s3_handler.get_available_models()
        selected = 0
        
        while True:
            # Clear screen
            self.screen.fill(self.BLACK)
            
            # Draw title
            title = self.title_font.render('Select Model', True, self.GOLD)
            title_rect = title.get_rect(center=(self.WIDTH/2, 100))
            self.screen.blit(title, title_rect)
            
            # Draw models list
            for i, model in enumerate(models):
                color = self.GOLD if i == selected else self.WHITE
                text = f"Model {model['checkpoint_number']} - Win Rate: {model['win_rate']:.2%}"
                text_surface = self.menu_font.render(text, True, color)
                self.screen.blit(text_surface, (100, 200 + i * 50))
                
                # If this is the selected model, show its training plot
                if i == selected and 'plot_s3_path' in model:
                    # Create temp directory if it doesn't exist
                    os.makedirs('temp', exist_ok=True)
                    plot_path = f"temp/plot_{model['checkpoint_number']}.png"
                    
                    # Download and display plot
                    try:
                        s3_handler.s3.download_file(
                            s3_handler.bucket_name,
                            model['plot_s3_path'],  # This is the path from DynamoDB
                            plot_path
                        )
                        plot_surface = pygame.image.load(plot_path)
                        plot_surface = pygame.transform.scale(plot_surface, (400, 200))
                        self.screen.blit(plot_surface, (350, 200))
                        os.remove(plot_path)  # Clean up temp file
                    except Exception as e:
                        print(f"Error loading plot: {e}")
            
            pygame.display.flip()
            
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.sounds['menu_move'].play()
                        selected = (selected - 1) % len(models)
                    elif event.key == pygame.K_DOWN:
                        self.sounds['menu_move'].play()
                        selected = (selected + 1) % len(models)
                    elif event.key == pygame.K_RETURN:
                        self.sounds['menu_select'].play()
                        # Download selected model
                        model = models[selected]
                        local_path = s3_handler.download_model(model['model_id'])
                        return local_path
                    elif event.key == pygame.K_ESCAPE:
                        return None

    def check_new_models(self):
        """Check SQS for new model notifications"""
        sqs = boto3.client('sqs')
        queue_url = "YOUR_SQS_QUEUE_URL"  # Get from terraform output
        
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10
        )
        
        if 'Messages' in response:
            for message in response['Messages']:
                print(f"New model available: {message['Body']}")
                # Delete message after processing
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )

if __name__ == "__main__":
    menu = Menu()
    menu.run() 