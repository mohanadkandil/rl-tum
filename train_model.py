import torch
import numpy as np
from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random
import argparse
import math
from cloud_storage import S3Handler
import time
from upload_model import upload_model

# Training parameters
EPISODES = 1000
EVAL_FREQUENCY = 50
EVAL_EPISODES = 50
BATCH_SIZE = 256
TARGET_UPDATE = 100
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0005
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998

def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to train (default: 200)')
    parser.add_argument('--eval-frequency', type=int, default=100,
                       help='How often to evaluate the agent (default: 100)')
    return parser.parse_args()

def evaluate_agent(agent, env):
    """Evaluate agent performance without exploration"""
    wins = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Turn off exploration
    
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        moves_made = 0
        
        while not done and moves_made < 100:
            env.player = -1  # AI plays as white
            valid_moves = env.valid_moves(env.player)
            
            if not valid_moves:
                break
                
            action = agent.act(state, valid_moves)
            next_state, reward, additional_moves, done = env.step(action, env.player)
            moves_made += 1
            
            if done:
                if env.game_winner(next_state) == -1:  # AI won
                    wins += 1
                break
                
            # Random opponent move
            env.player = 1
            valid_moves = env.valid_moves(env.player)
            
            if not valid_moves:
                wins += 1  # AI won if opponent has no moves
                break
                
            action = random.choice(valid_moves)
            next_state, reward, _, done = env.step(action, env.player)
            moves_made += 1
            
            if done:
                if env.game_winner(next_state) == -1:  # AI won
                    wins += 1
                break
                
            state = next_state
    
    agent.epsilon = original_epsilon
    return wins / EVAL_EPISODES

def train_agent(episodes=None):
    # Use arguments if episodes not specified
    if episodes is None:
        args = parse_args()
        episodes = args.episodes
        eval_frequency = args.eval_frequency
    else:
        eval_frequency = 50  # Default if episodes directly specified
    
    env = checkers_env()
    agent = DQNAgent()
    rewards = []
    win_rates = []
    total_steps = 0
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Starting training for {episodes} episodes...")
    best_win_rate = 0
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        moves_made = 0
        
        while moves_made < 200:  # Prevent infinite games
            # AI turn
            env.player = -1
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                break
                
            action = agent.act(state, valid_moves)
            next_state, reward, additional_moves, done = env.step(action, env.player)
            moves_made += 1
            total_steps += 1
            
            # Store transition
            agent.remember(state, action, reward, next_state, done)
            
            # Train more frequently
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                # Decay epsilon after each replay (more gradual)
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            if done:
                break
                
            # Opponent turn
            env.player = 1
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                break
                
            action = random.choice(valid_moves)
            next_state, reward, _, done = env.step(action, env.player)
            moves_made += 1
            
            if done:
                break
                
            state = next_state
            episode_reward += reward
            
            # Update target network periodically
            if total_steps % 1000 == 0:  # More frequent updates
                agent.update_target_network()
        
        rewards.append(episode_reward)
        
        # Evaluate periodically
        if episode % eval_frequency == 0:
            win_rate = evaluate_agent(agent, env)
            win_rates.append(win_rate)
            print(f"Episode {episode}/{episodes}, Win Rate: {win_rate:.2%}, "
                  f"Epsilon: {agent.epsilon:.3f}, Reward: {episode_reward:.1f}")
            
            # Save and upload best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                print(f"New best win rate: {win_rate:.2%}!")
                
                # Save model checkpoint
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'win_rate': win_rate,
                    'epsilon': agent.epsilon
                }, best_model_path)
                
                # Save training plot
                plot_path = os.path.join(checkpoint_dir, 'training_results.png')
                plot_training_results(rewards, win_rates, eval_frequency, save_path=plot_path)
                
                # Upload to API
                try:
                    result = upload_model(best_model_path, plot_path)
                    print(f"Upload successful: {result}")
                except Exception as e:
                    print(f"Upload failed: {e}")
    
    # Save and upload final model
    final_model_path = os.path.join(checkpoint_dir, f'model_final_e{episodes}.pth')
    torch.save({
        'episode': episodes,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'win_rate': win_rates[-1],
        'epsilon': agent.epsilon
    }, final_model_path)
    
    # Upload final model
    s3_handler = S3Handler()
    s3_handler.upload_training(
        checkpoint_path=final_model_path,
        plot_path=plot_path,
        training_info={
            'win_rate': win_rates[-1],
            'episodes': episodes,
            'final_reward': rewards[-1],
            'duration': time.time() - start_time,
            'is_final': True
        }
    )
    
    print(f"Training complete! Best win rate: {best_win_rate:.2%}")
    print(f"Models saved in {checkpoint_dir}/")
    
    return agent, rewards, win_rates, eval_frequency

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
        print(f"Model epsilon: {agent.epsilon:.3f}")
    else:
        print(f"No model found at {model_path}")
    
    return agent

if __name__ == "__main__":
    # Train new model
    trained_agent, rewards, win_rates, eval_frequency = train_agent()