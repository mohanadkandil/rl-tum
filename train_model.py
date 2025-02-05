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


class CheckersTrainer:
    def __init__(self, env, agent1, agent2, episodes=1000, eval_frequency=100):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = episodes
        self.eval_frequency = eval_frequency
        self.rewards = []
        self.win_rates = []

    def train(self):
        wins = {1: 0, -1: 0, 0: 0}

        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            current_agent = self.agent1 if random.random() < 0.5 else self.agent2
            opponent_agent = self.agent2 if current_agent == self.agent1 else self.agent1
            player = 1 if current_agent == self.agent1 else -1
            total_reward = 0

            while not done:
                valid_moves = self.env.valid_moves(player)
                if not valid_moves:
                    break

                action = current_agent.act(state, valid_moves)
                next_state, reward, additional_moves, done = self.env.step(action, player)

                current_agent.remember(state, action, reward, next_state, done)

                if len(current_agent.memory) > current_agent.batch_size:
                    current_agent.replay()
                    if current_agent.epsilon > current_agent.epsilon_min:
                        current_agent.epsilon = max(current_agent.epsilon_min, current_agent.epsilon * current_agent.epsilon_decay)


                state = next_state
                total_reward += reward

                if not additional_moves:
                    current_agent, opponent_agent = opponent_agent, current_agent
                    player *= -1

            self.agent1.update_target_network()
            self.agent2.update_target_network()
            self.rewards.append(total_reward)

            winner = self.env.game_winner(state)
            wins[winner] += 1

            if (episode + 1) % self.eval_frequency == 0:
                win_rate = (wins[1] / max(1, (episode + 1))) * 100
                self.win_rates.append(win_rate)
                print(f"Episode {episode + 1}: Win rate {win_rate:.2f}%")

        return self.agent1, self.agent2, self.rewards, self.win_rates, self.eval_frequency

    def evaluate(self, games=100):
        wins = {1: 0, -1: 0, 0: 0}

        for _ in range(games):
            state = self.env.reset()
            done = False
            player = 1

            while not done:
                agent = self.agent1 if player == 1 else self.agent2
                valid_moves = self.env.valid_moves(player)

                if not valid_moves:
                    break

                action = agent.act(state, valid_moves)
                state, _, additional_moves, done = self.env.step(action, player)

                if not additional_moves:
                    player *= -1

            winner = self.env.game_winner(state)
            wins[winner] += 1

        print(f"Evaluation Results: Agent 1 Wins: {wins[1]}, Agent 2 Wins: {wins[-1]}, Draws: {wins[0]}")

    def plot_training_results(self, save_path="training_results.png"):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards, label='Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Training Rewards')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(self.eval_frequency, len(self.win_rates) * self.eval_frequency + 1, self.eval_frequency),
                 self.win_rates, label='Win Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate Progression')
        plt.legend()

        plt.savefig(save_path)
        plt.show()

    def save_model(self, path="checkers_agent.pth"):
        torch.save({
            'agent1_state_dict': self.agent1.q_network.state_dict(),
            'agent2_state_dict': self.agent2.q_network.state_dict()
        }, path)
        print(f"Model saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to train (default: 200)')
    parser.add_argument('--eval-frequency', type=int, default=100,
                        help='How often to evaluate the agent (default: 100)')
    return parser.parse_args()

def train_agent(episodes=None):
    env = checkers_env()
    agent = DQNAgent()
    agent2 = DQNAgent()
    checkersTrainer = CheckersTrainer(env, agent, agent2)
    return checkersTrainer.train()


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
    trained_agent, rewards, win_rates, eval_frequency = train_agent()
