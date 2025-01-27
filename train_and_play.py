import numpy as np
from checkers_env import checkers_env
from DQNAgent import DQNAgent  # Changed from LearningAgent
import matplotlib.pyplot as plt
import torch
import random

# Training parameters
EPISODES = 200         
EVAL_FREQUENCY = 50     
EVAL_EPISODES = 50      
BATCH_SIZE = 64         
TARGET_UPDATE = 100    
MEMORY_SIZE = 10000    
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997  

def evaluate_agent(agent, env):
    """Evaluate agent performance without exploration"""
    wins = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0  
    
    for episode in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        moves_made = 0
        
        while not done and moves_made < 100:  
            # Switch players
            env.player = -1  
            valid_moves = env.valid_moves(env.player)
            
            if not valid_moves:
                wins += 1  # AI lost (no moves)
                break
                
            # AI move
            action = agent.act(state, valid_moves)
            state, reward, additional_moves = env.step(action, env.player)
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
                
            # Random opponent strategy
            action = random.choice(valid_moves)
            state, reward, additional_moves = env.step(action, env.player)
            moves_made += 1
            
            if env.game_winner(env.board) != 0:
                if env.game_winner(env.board) == -1:  # AI won
                    wins += 1
                break
    
    agent.epsilon = original_epsilon
    return wins / EVAL_EPISODES

def train_agent():
    env = checkers_env(board_size=6)  # Specify 6x6 board
    # Initialize DQN agent with proper parameters
    agent = DQNAgent(
        state_size=36,     # 6x6 board flattened
        action_size=1296,  # All possible moves (6x6x6x6)
        hidden_size=128
    )
    
    episode_rewards = []
    win_rates = []
    
    print("Starting training...")
    print(f"Using device: {agent.device}")
    print(f"Training for {EPISODES} episodes")
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            done = False
            moves_made = 0
            
            # Print episode start
            if episode % 10 == 0:
                print(f"\rEpisode {episode}/{EPISODES}", end="", flush=True)
            
            while not done and moves_made < 200:  # Move limit to prevent infinite games
                current_state = env.board.copy()
                valid_moves = env.valid_moves(env.player)
                
                if not valid_moves:
                    break
                    
                # Get action from DQN
                action = agent.act(current_state, valid_moves)
                
                # Take action
                next_state, reward, additional_moves = env.step(action, env.player)
                
                # Store transition in memory
                done = env.game_winner(env.board) != 0
                agent.remember(current_state, action, reward, next_state, done)
                
                # Train on a batch of memories
                if len(agent.memory) >= agent.batch_size:
                    agent.replay()
                
                total_reward += reward
                moves_made += 1
                
                # Handle additional captures
                while additional_moves and not done:
                    action = additional_moves[0]
                    next_state, reward, additional_moves = env.step(action, env.player)
                    total_reward += reward
                    moves_made += 1
                    done = env.game_winner(env.board) != 0
                    
                    if done:
                        break
            
            episode_rewards.append(total_reward)
            
            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()
                print(f"\nUpdating target network at episode {episode}")
            
            # Evaluate periodically
            if (episode + 1) % EVAL_FREQUENCY == 0:
                win_rate = evaluate_agent(agent, env)
                win_rates.append(win_rate)
                print(f"\nEpisode {episode + 1}/{EPISODES}")
                print(f"Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Moves per game: {moves_made}")
                print(f"Memory size: {len(agent.memory)}")
                print("------------------------")
                
                # Save model checkpoint
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'win_rate': win_rate,
                }, f'checkpoints/model_ep{episode+1}.pt')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return agent, episode_rewards, win_rates
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
    
    return agent, episode_rewards, win_rates

def plot_training_results(rewards, win_rates):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot win rates
    eval_episodes = np.arange(EVAL_FREQUENCY, EPISODES + 1, EVAL_FREQUENCY)
    ax2.plot(eval_episodes, win_rates)
    ax2.set_title('Agent Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    
    plt.tight_layout()
    plt.show()

def play_against_ai(agent):
    from gui import CheckersGUI
    game = CheckersGUI(board_size=6)  # Specify 6x6 board
    game.env.player = 1  # Human plays as red (1)
    
    # Override the make_move method to include AI moves
    original_make_move = game.make_move
    def new_make_move(move):
        if game.game_over:
            return
            
        # Human move
        original_make_move(move)
        
        if not game.game_over:
            # AI move
            game.env.player = -1
            valid_moves = game.env.valid_moves(-1)
            
            if valid_moves:
                # Use DQN to select move
                state = game.env.board.copy()
                ai_action = agent.act(state, valid_moves)
                board, reward, additional_moves = game.env.step(ai_action, -1)
                game.player_turn = 1
                game.env.player = 1
                
                # Handle additional captures
                while additional_moves and not game.game_over:
                    if len(additional_moves) > 0:
                        ai_action = additional_moves[0]
                        board, reward, additional_moves = game.env.step(ai_action, -1)
                
                winner = game.env.game_winner(board)
                if winner != 0:
                    game.game_over = True
                    game.winner = winner
                    print(f"Player {winner} wins!")
            else:
                print("AI has no valid moves")
                game.game_over = True
                game.winner = 1
                game.env.render()
    
    game.make_move = new_make_move
    game.run()

if __name__ == "__main__":
    # Create checkpoints directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the agent
    trained_agent, rewards, win_rates = train_agent()
    
    # Plot results
    plot_training_results(rewards, win_rates)
    
    # Save final model
    torch.save({
        'model_state_dict': trained_agent.q_network.state_dict(),
        'optimizer_state_dict': trained_agent.optimizer.state_dict(),
    }, 'checkpoints/final_model.pt')
    
    # Play against the trained agent
    play_against_ai(trained_agent) 