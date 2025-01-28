import torch
import numpy as np
from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random
import argparse

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

def train_agent(episodes=2000):
    env = checkers_env()
    agent = DQNAgent()
    rewards = []
    win_rates = []
    eval_frequency = 50
    total_steps = 0
    
    print("Starting training...")
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
            print(f"Episode {episode}, Win Rate: {win_rate:.2%}, Epsilon: {agent.epsilon:.3f}")

    return agent, rewards, win_rates, episodes, eval_frequency

def plot_training_results(rewards, win_rates, episodes, eval_frequency):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot win rates
    plt.subplot(1, 2, 2)
    eval_episodes = range(0, episodes + 1, eval_frequency)
    plt.plot(eval_episodes, win_rates)
    plt.title('Agent Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    trained_agent, rewards, win_rates, episodes, eval_frequency = train_agent()
    plot_training_results(rewards, win_rates, episodes, eval_frequency) 