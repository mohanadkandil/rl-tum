import time

import torch
import numpy as np
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import plotext as tplt
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to train (default: 200)')
    parser.add_argument('--eval-frequency', type=int, default=10,
                        help='How often to evaluate the agent (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for training (default: 0.0001)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for replay (default: 128)')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Number of games to play during evaluation (default: 100)')
    return parser.parse_args()


class CheckersTrainer:
    def __init__(self, env, agent1, agent2, args):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = args.episodes
        self.eval_frequency = args.eval_frequency
        self.eval_games = args.eval_games
        self.rewards = []
        self.win_rates = []

    def train(self):
        wins = {1: 0, -1: 0, 0: 0}
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        console = Console()
        with Live(console=console, refresh_per_second=2) as live:
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

                    # Encourage positional play
                    if player == 1:
                        reward += 0.1 * (action[2] - action[0])
                    else:
                        reward += 0.1 * (action[0] - action[2])

                    current_agent.remember(state, action, reward, next_state, done)

                    if len(current_agent.memory) > current_agent.batch_size:
                        current_agent.replay()
                        if current_agent.epsilon > current_agent.epsilon_min:
                            current_agent.epsilon = max(current_agent.epsilon_min,
                                                        current_agent.epsilon * current_agent.epsilon_decay)

                    state = next_state
                    total_reward += reward

                    if not additional_moves:
                        current_agent, opponent_agent = opponent_agent, current_agent
                        player *= -1

                if episode % 10 == 0:
                    self.agent1.update_target_network()
                    self.agent2.update_target_network()

                self.rewards.append(total_reward)

                winner = self.env.game_winner(state)
                wins[winner] += 1

                if (episode + 1) % self.eval_frequency == 0:
                    self.evaluate()
                    win_rate = (wins[1] / max(1, (episode + 1))) * 100
                    self.win_rates.append(win_rate)
                    plot = self.live_plot_terminal()
                    live.update(Panel(plot, title="Win Rate"))
                    print(f"Episode {episode + 1}: Win rate {win_rate:.2f}%")

        self.save_model(os.path.join(checkpoint_dir, f'model_final_e{self.episodes}.pth'))
        plot_path = os.path.join(checkpoint_dir, 'training_results.png')
        self.plot_training_results(plot_path)

        return self.agent1, self.agent2, self.rewards, self.win_rates, self.eval_frequency

    def evaluate(self):
        wins = {1: 0, -1: 0, 0: 0}

        for _ in range(self.eval_games):
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

    def live_plot_terminal(self):
        """Creates a live-updating plot using plotext."""
        """Creates a live-updating plot using plotext with better scaling and readability."""
        #tplt.clf()  # Clear previous plot

        tplt.xlabel("Episodes")
        tplt.ylabel("Win Rate (%)")

        eval_episodes = range(0, len(self.rewards), self.eval_frequency)
        if len(eval_episodes) > len(self.win_rates):
            eval_episodes = eval_episodes[:len(self.win_rates)]
        tplt.plot(eval_episodes, self.win_rates)
        tplt.ylim(30, 100)  # Keep Y-axis fixed between 30% and 100%


        return tplt.build()


    def plot_training_results(self, save_path='training_results.png'):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards, label='Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Training Rewards')
        plt.legend()

        plt.subplot(1, 2, 2)
        eval_episodes = range(0, len(self.rewards), self.eval_frequency)
        if len(eval_episodes) > len(self.win_rates):
            eval_episodes = eval_episodes[:len(self.win_rates)]
        plt.plot(eval_episodes, self.win_rates)
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate Progression')
        plt.legend()

        plt.savefig(save_path)
        plt.show()

    def save_model(self, path="checkers_agent.pth"):
        """Save the best performing agent based on win rate."""
        best_agent = self.agent1 if self.win_rates[-1] > self.win_rates[-2] else self.agent2
        torch.save({
            'episode': self.episodes,
            'model_state_dict': best_agent.q_network.state_dict(),
            'optimizer_state_dict': best_agent.optimizer.state_dict(),
            'win_rate': self.win_rates[-1],
            'epsilon': best_agent.epsilon
        }, path)
        print(f"Best model saved to {path} with win rate: {self.win_rates[-1]:.2f}%")
        print(f"Model saved to {path}")


def train_agent(episodes=None):
    env = checkers_env()
    agent = DQNAgent()
    agent2 = DQNAgent()
    checkersTrainer = CheckersTrainer(env, agent, agent2, args=parse_args())
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
    train_agent()
