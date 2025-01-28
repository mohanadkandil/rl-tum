import torch
import numpy as np
from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random

# Training parameters
EPISODES = 200
EVAL_FREQUENCY = 100
EVAL_EPISODES = 30
BATCH_SIZE = 128
TARGET_UPDATE = 200
MEMORY_SIZE = 50000
LEARNING_RATE = 0.0005
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998

def evaluate_agent(agent, env):
    """Evaluate agent performance without exploration"""
    wins = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Turn off exploration
    
    for episode in range(EVAL_EPISODES):
        env.reset()  # Reset environment
        done = False
        moves_made = 0
        
        while not done and moves_made < 100:
            env.player = -1  # AI plays as white
            valid_moves = env.valid_moves(env.player)
            
            if not valid_moves:
                wins += 1  # AI lost (no moves)
                break
                
            # Get current board state
            current_state = env.board.copy()  # Make sure we're using the board state
            action = agent.act(current_state, valid_moves)
            next_state, reward, additional_moves = env.step(action, env.player)
            moves_made += 1
            
            if env.game_winner(env.board) == 1:
                wins += 1
                break
            elif env.game_winner(env.board) == -1:
                break
                
            # Random opponent move
            env.player = 1
            valid_moves = env.valid_moves(env.player)
            
            if not valid_moves:
                break
                
            action = random.choice(valid_moves)
            next_state, reward, additional_moves = env.step(action, env.player)
            moves_made += 1
            
            if env.game_winner(env.board) != 0:
                if env.game_winner(env.board) == -1:
                    wins += 1
                break
    
    agent.epsilon = original_epsilon
    return wins / EVAL_EPISODES

def train_agent():
    env = checkers_env()
    agent = DQNAgent(
        state_size=36,
        action_size=1296,
        hidden_size=256
    )
    
    episode_rewards = []
    win_rates = []
    
    print(f"Using device: {agent.device}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            done = False
            moves_made = 0
            
            if episode % 10 == 0:
                print(f"\rEpisode {episode}/{EPISODES}", end="", flush=True)
            
            while not done and moves_made < 200:
                current_state = env.board.copy()
                valid_moves = env.valid_moves(env.player)
                
                if not valid_moves:
                    break
                    
                action = agent.act(current_state, valid_moves)
                next_state, reward, additional_moves = env.step(action, env.player)
                done = env.game_winner(env.board) != 0
                agent.remember(current_state, action, reward, next_state, done)
                
                if len(agent.memory) >= agent.batch_size:
                    agent.replay()
                
                total_reward += reward
                moves_made += 1
                
                while additional_moves and not done:
                    action = additional_moves[0]
                    next_state, reward, additional_moves = env.step(action, env.player)
                    total_reward += reward
                    moves_made += 1
                    done = env.game_winner(env.board) != 0
            
            episode_rewards.append(total_reward)
            
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()
                print(f"\nUpdating target network at episode {episode}")
            
            if (episode + 1) % EVAL_FREQUENCY == 0:
                win_rate = evaluate_agent(agent, env)
                win_rates.append(win_rate)
                print(f"\nEpisode {episode + 1}/{EPISODES}")
                print(f"Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Memory size: {len(agent.memory)}")
                print("------------------------")
                
                # Save checkpoint
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'win_rate': win_rate,
                }, f'checkpoints/model_ep{episode+1}.pt')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, 'checkpoints/final_model.pt')
    
    return agent, episode_rewards, win_rates

def plot_training_results(rewards, win_rates):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    eval_episodes = np.arange(EVAL_FREQUENCY, EPISODES + 1, EVAL_FREQUENCY)
    ax2.plot(eval_episodes, win_rates)
    ax2.set_title('Agent Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    trained_agent, rewards, win_rates = train_agent()
    plot_training_results(rewards, win_rates) 