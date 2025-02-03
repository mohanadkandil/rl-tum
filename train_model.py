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

def evaluate_agent(agent, env, num_games=50, opponent_strategy=None):
    """Evaluate agent performance"""
    if opponent_strategy is None:
        opponent_strategy = select_strategic_move
        
    wins = 0
    draws = 0
    total_moves = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Small exploration during evaluation
    
    for game in range(num_games):
        state = env.reset()
        moves = 0
        max_moves = 100
        game_over = False
        
        while not game_over and moves < max_moves:
            # AI's turn (player -1)
            valid_moves = env.valid_moves(-1)  # AI plays as -1
            if not valid_moves:
                # AI has no moves, it loses
                game_over = True
                break
                
            action = agent.act(state, valid_moves)
            next_state, reward, additional_moves, done = env.step(action, -1)
            state = next_state
            
            if done:
                if env.game_winner(state) == -1:
                    wins += 1
                game_over = True
                break
                
            # Opponent's turn (player 1)
            valid_moves = env.valid_moves(1)
            if not valid_moves:
                # Opponent has no moves, AI wins
                wins += 1
                game_over = True
                break
                
            action = opponent_strategy(env, valid_moves)
            next_state, reward, additional_moves, done = env.step(action, 1)
            state = next_state
            
            if done:
                if env.game_winner(state) == -1:
                    wins += 1
                game_over = True
                break
                
            moves += 1
            
        if moves >= max_moves:
            draws += 1
            
        total_moves += moves
    
    agent.epsilon = original_epsilon
    
    # Calculate metrics
    win_rate = wins / num_games
    draw_rate = draws / num_games
    avg_moves = total_moves / max(1, num_games)
    
    print(f"Debug - Games: {num_games}, Wins: {wins}, Draws: {draws}")  # Add debug info
    
    return win_rate, draw_rate, avg_moves, 0, 0

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

def train_episode(agent, env, phase):
    """Train for one episode with specified opponent phase"""
    if phase not in {"random", "basic", "medium", "advanced"}:
        print(f"Invalid phase {phase}, defaulting to random")
        phase = "random"
        
    state = env.reset()
    total_reward = 0
    done = False
    moves = 0
    max_moves = 100
    
    opponent_strategies = {
        "random": select_random_move,
        "basic": select_basic_move,
        "medium": select_strategic_move,
        "advanced": select_advanced_move
    }
    
    try:
        opponent_strategy = opponent_strategies[phase]
        
        while not done and moves < max_moves:
            # AI's turn
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                break
                
            try:
                action = agent.act(state, valid_moves)
                next_state, reward, additional_moves, done = env.step(action, env.player)
                
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                
                # Handle additional captures
                while additional_moves and not done:
                    action = additional_moves[0]
                    next_state, reward, additional_moves, done = env.step(action, env.player)
                    agent.remember(state, action, reward, next_state, done)
                    total_reward += reward
                
                if done:
                    break
                    
                # Opponent's turn
                valid_moves = env.valid_moves(-env.player)
                if not valid_moves:
                    break
                    
                opponent_action = opponent_strategy(env, valid_moves)
                state, _, additional_moves, done = env.step(opponent_action, -env.player)
                
                while additional_moves and not done:
                    opponent_action = opponent_strategy(env, additional_moves)
                    state, _, additional_moves, done = env.step(opponent_action, -env.player)
                
                moves += 1
                
            except Exception as e:
                print(f"Error during move: {e}")
                break
        
        # Train if enough experiences
        if len(agent.memory) >= agent.batch_size:
            try:
                agent.replay()
                if moves % TARGET_UPDATE == 0:
                    agent.update_target_network()
            except Exception as e:
                print(f"Error during training: {e}")
                
    except Exception as e:
        print(f"Error during episode: {e}")
        return 0
        
    return total_reward

def save_model(agent, episode, metrics):
    """Save model with metrics"""
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'metrics': metrics,
    }, f'checkpoints/model_ep{episode}.pth')
    
    # Also save as best model if it's the best so far
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'metrics': metrics,
    }, 'checkpoints/best_model.pth')

def train_agent():
    args = parse_args()
    episodes = args.episodes
    
    # Add validation tracking
    validation_scores = []
    best_val_score = float('-inf')
    patience = 10
    patience_counter = 0
    
    # Add model snapshots
    model_snapshots = []
    snapshot_frequency = episodes // 10  # Save 10 snapshots during training
    
    try:
        env = checkers_env()
        agent = DQNAgent()
        rewards = []
        metrics = []
        
        # Training phases with validation
        phase_ratio = episodes / 5000
        curriculum_phases = [
            (0, "random", 0.3),  # Phase, opponent, target win rate
            (int(1000 * phase_ratio), "basic", 0.5),
            (int(2000 * phase_ratio), "medium", 0.6),
            (int(3000 * phase_ratio), "advanced", 0.7)
        ]
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            # Get current phase and target
            current_phase = next((phase for phase in curriculum_phases 
                                if episode >= phase[0]), curriculum_phases[-1])
            phase_name = current_phase[1]
            target_win_rate = current_phase[2]
            
            # Train episode
            episode_reward = train_episode(agent, env, phase_name)
            rewards.append(episode_reward)
            
            # Regular evaluation
            if episode % EVAL_FREQUENCY == 0:
                metrics = evaluate_with_different_opponents(agent, env)
                win_rate = metrics['average_win_rate']
                
                # Validation score combines win rate and target achievement
                val_score = win_rate - abs(win_rate - target_win_rate)
                validation_scores.append(val_score)
                
                print(f"\nEpisode {episode}/{episodes}")
                print(f"Phase: {phase_name} (target: {target_win_rate:.1%})")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Validation Score: {val_score:.2f}")
                
                # Early stopping with validation score
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    save_model(agent, episode, metrics)
                else:
                    patience_counter += 1
                
                # Take model snapshot
                if episode % snapshot_frequency == 0:
                    model_snapshots.append({
                        'episode': episode,
                        'state_dict': agent.q_network.state_dict().copy(),
                        'metrics': metrics.copy()
                    })
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping at episode {episode}")
                    print("Selecting best model from snapshots...")
                    best_snapshot = max(model_snapshots, 
                                     key=lambda x: x['metrics']['average_win_rate'])
                    agent.q_network.load_state_dict(best_snapshot['state_dict'])
                    break
            
            # Dynamic epsilon decay based on performance
            if agent.epsilon > EPSILON_END:
                decay_rate = EPSILON_DECAY
                if len(validation_scores) > 1:
                    # Adjust decay based on recent performance
                    if validation_scores[-1] < validation_scores[-2]:
                        decay_rate = decay_rate * 0.9  # Slower decay if performance drops
                agent.epsilon = max(EPSILON_END, 
                                 agent.epsilon * decay_rate)
        
        return agent, rewards, metrics
    
    except Exception as e:
        print(f"Training error: {e}")
        return None, [], []

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