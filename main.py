import checkers_env
import LearningAgent
import random
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
EPISODES = 1000        # Total number of games to play
LEARNING_RATE = 0.1    # How quickly the agent learns (step_size)
EPSILON = 0.1          # Probability of choosing random actions (exploration)
GAMMA = 0.99          # Discount factor for future rewards

def train_agent():
    env = checkers_env.checkers_env()
    agent = LearningAgent.LearningAgent(step_size=LEARNING_RATE, epsilon=EPSILON, env=env)
    
    episode_rewards = []
    
    for episode in range(EPISODES):
        env.reset()
        total_reward = 0
        done = False
        
        while not done:
            current_state = env.board.copy()
            action = agent.select_action()
            
            if action is None:
                break
                
            next_state, reward, additional_moves = env.step(action, env.player)
            agent.learning(current_state, action, reward, next_state)
            
            total_reward += reward
            
            if env.game_winner(env.board) != 0:
                done = True
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward}")
    
    return episode_rewards

def plot_training_results(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == "__main__":
    # Train the agent
    rewards = train_agent()
    
    # Plot results
    plot_training_results(rewards)



