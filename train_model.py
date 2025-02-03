import torch
import numpy as np
from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random
import argparse
import math

# Training parameters
EPISODES = 5000
EVAL_FREQUENCY = 25  # More frequent evaluation
EVAL_EPISODES = 100  # More evaluation games
BATCH_SIZE = 128  # Smaller batch size
TARGET_UPDATE = 50  # More frequent target updates
MEMORY_SIZE = 50000  # Smaller memory to focus on recent experiences
LEARNING_RATE = 0.0001  # Smaller learning rate for stability
EPSILON_START = 1.0
EPSILON_END = 0.1  # Higher minimum exploration
EPSILON_DECAY = 0.999  # Slower decay

def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to train (default: 200)')
    parser.add_argument('--eval-frequency', type=int, default=100,
                       help='How often to evaluate the agent (default: 100)')
    return parser.parse_args()

def evaluate_agent(agent, env, num_games=50):  # Increased from 20 to 50 for better evaluation
    """Evaluate agent performance"""
    wins = 0
    draws = 0
    total_moves = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Small exploration during evaluation
    
    # Track more metrics
    captures = 0
    king_promotions = 0
    
    for game in range(num_games):
        state = env.reset()
        moves = 0
        max_moves = 100
        
        while moves < max_moves:
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                break
                
            if env.player == -1:  # AI's turn
                action = agent.act(state, valid_moves)
                next_state, reward, additional_moves, done = env.step(action, env.player)
                
                # Track metrics
                if abs(action[0] - action[2]) == 2:  # Capture move
                    captures += 1
                if next_state[action[2]][action[3]] == -2:  # King promotion
                    king_promotions += 1
                    
            else:  # Opponent's turn - use deterministic strategy
                action = select_strategic_move(env, valid_moves)
                next_state, reward, additional_moves, done = env.step(action, env.player)
            
            state = next_state
            moves += 1
            
            if done:
                if env.game_winner(state) == -1:
                    wins += 1
                break
        
        if moves >= max_moves:
            draws += 1
        total_moves += moves
    
    agent.epsilon = original_epsilon
    win_rate = wins / num_games
    draw_rate = draws / num_games
    avg_moves = total_moves / num_games
    avg_captures = captures / num_games
    avg_promotions = king_promotions / num_games
    
    return win_rate, draw_rate, avg_moves, avg_captures, avg_promotions

def select_strategic_move(env, valid_moves):
    """Deterministic opponent strategy"""
    # Prioritize moves
    capture_moves = [m for m in valid_moves if abs(m[0] - m[2]) == 2]
    king_moves = [m for m in valid_moves if abs(env.board[m[0]][m[1]]) == 2]
    forward_moves = [m for m in valid_moves if m[2] > m[0]]  # Moving forward
    
    if capture_moves:
        return capture_moves[0]  # Always take first capture
    elif king_moves:
        return king_moves[0]  # Use kings strategically
    elif forward_moves:
        return forward_moves[0]  # Move forward when possible
    return valid_moves[0]  # Take first available move

def select_random_move(env, valid_moves):
    """Completely random opponent"""
    return random.choice(valid_moves)

def select_basic_move(env, valid_moves):
    """Basic opponent - only prefers captures"""
    captures = [m for m in valid_moves if abs(m[0] - m[2]) == 2]
    return random.choice(captures) if captures else random.choice(valid_moves)

def select_advanced_move(env, valid_moves):
    """Advanced opponent - uses position evaluation"""
    best_score = float('-inf')
    best_move = valid_moves[0]
    
    for move in valid_moves:
        # Make temporary move
        temp_board = env.board.copy()
        env.board[move[2]][move[3]] = env.board[move[0]][move[1]]
        env.board[move[0]][move[1]] = 0
        
        # Evaluate position
        score = evaluate_position(env.board, env.player)
        
        # Restore board
        env.board = temp_board
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

def evaluate_position(board, player):
    """Evaluate board position"""
    score = 0
    
    # Piece values
    piece_values = {
        1: 100,   # Regular piece
        2: 250    # King
    }
    
    # Position values (center control, back rank protection)
    position_values = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 2, 2, 1, 1],
        [1, 2, 3, 3, 2, 1],
        [1, 2, 3, 3, 2, 1],
        [1, 1, 2, 2, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    
    for row in range(6):
        for col in range(6):
            piece = board[row][col]
            if piece != 0:
                # Base piece value
                value = piece_values[abs(piece)]
                
                # Position value
                value += position_values[row][col] * 10
                
                # Back rank bonus
                if (piece > 0 and row == 0) or (piece < 0 and row == 5):
                    value += 20
                
                score += value if piece * player > 0 else -value
    
    return score

def train_agent(episodes=5000):
    env = checkers_env()
    agent = DQNAgent()
    rewards = []
    metrics = []
    
    # Early stopping parameters
    patience = 15  # Increased patience
    min_improvement = 0.02  # Minimum improvement threshold
    best_win_rate = 0
    no_improvement_count = 0
    
    # Curriculum learning - start with simpler opponent
    curriculum_phases = [
        (0, "random"),      # First 1000 episodes
        (1000, "basic"),    # Next 1000 episodes
        (2000, "medium"),   # Next 1000 episodes
        (3000, "advanced")  # Rest of training
    ]
    
    print("Starting training...")
    
    for episode in range(episodes):
        # Update curriculum phase
        current_phase = next(phase[1] for phase in curriculum_phases 
                           if episode >= phase[0])
        
        # Train episode
        episode_reward = train_episode(agent, env, current_phase)
        rewards.append(episode_reward)
        
        # Evaluate periodically
        if episode % EVAL_FREQUENCY == 0:
            metrics = evaluate_with_different_opponents(agent, env)
            win_rate = metrics['average_win_rate']
            
            print(f"\nEpisode {episode}/{episodes}")
            print(f"Phase: {current_phase}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Win Rates by Opponent:")
            for opp, rate in metrics['opponent_win_rates'].items():
                print(f"  {opp}: {rate:.2%}")
            
            # Early stopping check with minimum improvement threshold
            if win_rate > best_win_rate + min_improvement:
                best_win_rate = win_rate
                no_improvement_count = 0
                save_model(agent, episode, metrics)
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                print(f"Early stopping at episode {episode}")
                print("No significant improvement for", patience, "evaluations")
                break
                
        # Decay epsilon more gradually
        if agent.epsilon > EPSILON_END:
            agent.epsilon = max(EPSILON_END, 
                              EPSILON_START * (EPSILON_DECAY ** episode))
    
    return agent, rewards, metrics

def evaluate_with_different_opponents(agent, env):
    """Evaluate against different opponent strategies"""
    opponents = {
        'random': select_random_move,
        'basic': select_basic_move,
        'medium': select_strategic_move,
        'advanced': select_advanced_move
    }
    
    metrics = {
        'opponent_win_rates': {},
        'average_win_rate': 0.0
    }
    
    for opp_name, opp_strategy in opponents.items():
        win_rate, _, _, _, _ = evaluate_agent(agent, env, 
                                            num_games=EVAL_EPISODES,
                                            opponent_strategy=opp_strategy)
        metrics['opponent_win_rates'][opp_name] = win_rate
    
    metrics['average_win_rate'] = np.mean(list(metrics['opponent_win_rates'].values()))
    return metrics

def plot_training_results(rewards, win_rates, eval_frequency=50, save_path='training_results.png'):
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(range(len(rewards)), rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot win rates
    plt.subplot(1, 2, 2)
    # Calculate correct x-axis points for win rates
    eval_episodes = range(0, len(rewards), eval_frequency)
    if len(eval_episodes) > len(win_rates):
        eval_episodes = eval_episodes[:len(win_rates)]
    plt.plot(eval_episodes, win_rates)
    plt.title('Agent Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_trained_model(model_path):
    """Load a trained model"""
    agent = DQNAgent()  
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=agent.device)
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        
        print(f"Loaded model from {model_path}")
        print(f"Model win rate: {checkpoint['win_rate']:.2%}")
        print(f"Model draw rate: {checkpoint['draw_rate']:.2%}")
        print(f"Model average moves: {checkpoint['avg_moves']:.2f}")
    else:
        print(f"No model found at {model_path}")
    
    return agent

if __name__ == "__main__":
    # Train new model
    trained_agent, rewards, metrics = train_agent()