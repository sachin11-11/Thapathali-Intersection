import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime

from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent
import config


def plot_rewards(episode_rewards, save_path):
    """
    Plot training rewards over episodes.
    
    Args:
        episode_rewards: List of total rewards per episode
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards per Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, label=f'{window_size}-Episode Moving Average', color='red', linewidth=2)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards with Moving Average')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reward plot saved to {save_path}")
    plt.close()


def train():
    """Main training loop for DQN traffic light control."""
    
    # Create directories for outputs
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize environment and agent
    print("Initializing SUMO environment...")
    env = SumoEnvironment(use_gui=config.SUMO_GUI)
    
    # Reset to get state size
    initial_state = env.reset()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Incoming lanes: {len(env.incoming_lanes)}")
    print(f"Outgoing lanes: {len(env.outgoing_lanes)}")
    
    # Initialize DQN agent
    print("Initializing DQN agent...")
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    start_episode = 0
    
    # Check for existing checkpoints to resume
    checkpoint_dir = "trained_models"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pth")]
    if checkpoints:
        # Sort by episode number
        checkpoints.sort(key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pth", "")))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"Found checkpoint: {latest_checkpoint}. Loading...")
        agent.load(checkpoint_path)
        start_episode = int(latest_checkpoint.replace("checkpoint_ep", "").replace(".pth", ""))
        print(f"Resuming from episode {start_episode + 1}...")
    
    print(f"\nStarting training for {config.EPISODES} episodes...")
    print("=" * 70)
    
    for episode in range(start_episode, config.EPISODES):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        step_count = 0
        
        done = False
        while not done:
            # Select action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network periodically
        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{config.EPISODES} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {step_count}")
        
        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            checkpoint_path = f"trained_models/checkpoint_ep{episode+1}.pth"
            agent.save(checkpoint_path)
    
    print("=" * 70)
    print("Training completed!")
    
    # Save final model
    agent.save(config.MODEL_SAVE_PATH)
    
    # Close environment
    env.close()
    
    # Plot and save rewards
    print("\nGenerating reward plot...")
    plot_rewards(episode_rewards, config.PLOT_SAVE_PATH)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Episodes: {config.EPISODES}")
    print(f"Average Reward (All): {np.mean(episode_rewards):.2f}")
    print(f"Average Reward (Last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print("=" * 70)
    
    return episode_rewards


if __name__ == "__main__":
    train()
