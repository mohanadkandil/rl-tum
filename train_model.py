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
EVAL_FREQUENCY = 25
EVAL_EPISODES = 100  # More evaluation games for stability
BATCH_SIZE = 128  # Larger batch size
TARGET_UPDATE = 10  # Update target network more frequently
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0001  # Lower learning rate for stability
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999  # Slower decay

def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train (default: 1000)')
    parser.add_argument('--eval-frequency', type=int, default=25,
                       help='How often to evaluate the agent (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
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
    
    max_moves = 200  # Increase max moves to allow for draws
    draw_threshold = 50  # Consider it a draw if no captures in this many moves
    last_capture_move = 0
    
    for game in range(num_games):
        state = env.reset()
        moves = 0
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
                
            # Check for draw conditions
            if moves - last_capture_move > draw_threshold:
                draws += 1
                game_over = True
                break
            
            # Update last_capture_move if a capture occurred
            if abs(reward) > 1:  # Assuming capture rewards are > 1
                last_capture_move = moves
            
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

def train_episode(agent, env, opponent_type):
    state = env.reset()
    total_reward = 0
    done = False
    moves_without_capture = 0
    
    while not done:
        valid_moves = env.valid_moves(env.player)
        if not valid_moves:
            break
            
        action = agent.act(state, valid_moves)
        next_state, reward, additional_moves, done = env.step(action, env.player)
        
        # Track moves without capture
        if abs(reward) > 1:  # Capture occurred
            moves_without_capture = 0
        else:
            moves_without_capture += 1
            if moves_without_capture > 50:  # Draw condition
                reward = 0  # Neutral reward for draws
                done = True
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        if len(agent.memory) % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        state = next_state
        total_reward += reward
    
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
    
    validation_scores = []
    win_rates = []
    rewards = []
    episode_rewards = []  # Track rewards per episode
    best_win_rate = 0
    moving_averages = []
    window_size = 10
    
    # Add experience replay warmup
    min_experiences = 1000
    
    try:
        env = checkers_env()
        agent = DQNAgent()
        
        # Modified curriculum - slower progression
        phase_ratio = episodes / 5000
        curriculum_phases = [
            (0, "random", 0.3),  # Start with lower target
            (int(2000 * phase_ratio), "random", 0.5),  # Stay on random longer
            (int(3000 * phase_ratio), "basic", 0.4),
            (int(4000 * phase_ratio), "medium", 0.3)
        ]
        
        print(f"Starting training for {episodes} episodes...")
        print("Collecting initial experiences...")
        
        # Collect initial experiences
        while len(agent.memory) < min_experiences:
            state = env.reset()
            done = False
            while not done:
                valid_moves = env.valid_moves(env.player)
                if not valid_moves:
                    break
                action = agent.act(state, valid_moves)
                next_state, reward, additional_moves, done = env.step(action, env.player)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
        
        print(f"Initial experiences collected: {len(agent.memory)}")
        
        for episode in range(episodes):
            current_phase = next((phase for phase in curriculum_phases 
                                if episode >= phase[0]), curriculum_phases[-1])
            phase_name = current_phase[1]
            target_win_rate = current_phase[2]
            
            episode_total_reward = 0
            # Multiple training steps per episode
            for _ in range(8):  # Double training frequency
                step_reward = train_episode(agent, env, phase_name)
                episode_total_reward += step_reward
            
            # Store average reward for the episode
            rewards.append(episode_total_reward / 8)
            
            if episode % args.eval_frequency == 0:
                metrics = evaluate_with_different_opponents(agent, env)
                win_rate = metrics['average_win_rate']
                win_rates.append(win_rate)
                
                # Calculate moving average
                if len(win_rates) >= window_size:
                    moving_avg = sum(win_rates[-window_size:]) / window_size
                    moving_averages.append(moving_avg)
                else:
                    moving_avg = win_rate
                    moving_averages.append(moving_avg)
                
                val_score = win_rate - abs(win_rate - target_win_rate)
                validation_scores.append(val_score)
                
                print(f"\nEpisode {episode}/{episodes}")
                print(f"Phase: {phase_name} (target: {target_win_rate:.1%})")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Moving Avg Win Rate: {moving_avg:.2%}")
                print(f"Validation Score: {val_score:.2f}")
                print("Win Rates by Opponent:")
                for opp, rate in metrics['opponent_win_rates'].items():
                    print(f"  {opp}: {rate:.2%}")
                
                # Save if current win rate is best
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    save_model(agent, episode, metrics)
            
            # Smarter epsilon decay
            if agent.epsilon > EPSILON_END:
                if len(moving_averages) > 1:
                    if moving_averages[-1] > moving_averages[-2]:
                        # Performing well, decay normally
                        decay_rate = EPSILON_DECAY
                    else:
                        # Not improving, decay slower
                        decay_rate = EPSILON_DECAY * 0.97
                else:
                    decay_rate = EPSILON_DECAY
                agent.epsilon = max(EPSILON_END, agent.epsilon * decay_rate)
        
        print(f"\nTraining completed.")
        print(f"Best win rate achieved: {best_win_rate:.2%}")
        
        # Plot with corrected data
        plot_training_results(rewards, win_rates, args.eval_frequency)
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
        'average_win_rate': 0.0,
        'games_played': {}
    }
    
    print("\nDetailed Evaluation:")
    print("-" * 40)
    
    for opp_name, opp_strategy in opponents.items():
        win_rate, draws, avg_moves, _, _ = evaluate_agent(agent, env, 
                                                      num_games=EVAL_EPISODES,
                                                      opponent_strategy=opp_strategy)
        metrics['opponent_win_rates'][opp_name] = win_rate
        metrics['games_played'][opp_name] = EVAL_EPISODES
        
        print(f"{opp_name.capitalize()} Opponent:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Draw Rate: {draws/EVAL_EPISODES:.2%}")
        print(f"  Avg Moves: {avg_moves:.1f}")
    
    metrics['average_win_rate'] = np.mean(list(metrics['opponent_win_rates'].values()))
    print("-" * 40)
    print(f"Overall Win Rate: {metrics['average_win_rate']:.2%}")
    
    return metrics

def plot_training_results(rewards, win_rates, eval_frequency):
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards (per episode)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    # Plot win rates - make sure x-axis matches number of evaluations
    plt.subplot(1, 2, 2)
    eval_episodes = range(0, len(rewards), eval_frequency)
    if len(eval_episodes) > len(win_rates):
        eval_episodes = eval_episodes[:len(win_rates)]
    plt.plot(eval_episodes, win_rates)
    plt.title('Agent Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
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