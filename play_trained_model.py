import torch
from DQNAgent import DQNAgent
from gui import CheckersGUI
import os

def load_trained_model(model_path='checkpoints/best_model.pth'):
    # Initialize agent without hidden_size parameter
    agent = DQNAgent(state_size=36, action_size=1296)
    
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            agent.target_network.load_state_dict(checkpoint['model_state_dict'])
            agent.epsilon = 0  # No exploration during play
            print("Model loaded successfully")
        else:
            print(f"No model found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return agent

if __name__ == "__main__":
    # Load the trained model
    agent = load_trained_model()
    print(f"Model loaded successfully. Using device: {agent.device}")
    
    # Start the GUI game
    game = CheckersGUI()
    # AI plays as red (1), human plays as white (-1)
    game.player_turn = -1  
    game.run(agent) 