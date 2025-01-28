import torch
from DQNAgent import DQNAgent
from gui import CheckersGUI

def load_trained_model(model_path='checkpoints/final_model.pt'):
    agent = DQNAgent(
        state_size=36,
        action_size=1296,
        hidden_size=256
    )
    
    checkpoint = torch.load(model_path, weights_only=True)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0  # No exploration during play
    
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