import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent
import config


def run_fixed_time_control(env, max_steps=500):
    """
    Run simulation with fixed-time traffic light control.
    Cycles through phases with fixed durations.
    
    Returns:
        rewards: List of rewards at each step
        step_rewards: Reward at each simulation second
    """
    state = env.reset()
    rewards = []
    step_rewards = []
    phase_duration = config.GREEN_PHASE_DURATION  # 15 seconds per phase
    current_phase = 0
    time_in_phase = 0
    
    for step in range(max_steps):
        # Fixed-time: switch phase every 15 seconds
        if time_in_phase >= phase_duration:
            current_phase = (current_phase + 1) % config.NUM_ACTIONS
            time_in_phase = 0
        
        # Execute action
        next_state, reward, done, info = env.step(current_phase)
        rewards.append(reward)
        
        # Track individual step rewards during the green phase
        for _ in range(config.ACTION_DURATION):
            step_reward = env._calculate_reward()
            step_rewards.append(step_reward)
        
        state = next_state
        time_in_phase += config.ACTION_DURATION
        
        if done:
            break
    
    return rewards, step_rewards


def run_dqn_control(env, agent, max_steps=500):
    """
    Run simulation with DQN-based traffic light control.
    
    Returns:
        rewards: List of rewards at each step
        step_rewards: Reward at each simulation second
    """
    state = env.reset()
    rewards = []
    step_rewards = []
    
    for step in range(max_steps):
        # DQN: select action based on learned policy
        action = agent.act(state, training=False)  # Use greedy policy
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # Track individual step rewards during the green phase
        for _ in range(config.ACTION_DURATION):
            step_reward = env._calculate_reward()
            step_rewards.append(step_reward)
        
        state = next_state
        
        if done:
            break
    
    return rewards, step_rewards


def plot_comparison(fixed_rewards, dqn_rewards, save_path):
    """
    Create comparison plots for fixed-time vs DQN control.
    
    Args:
        fixed_rewards: Rewards from fixed-time control
        dqn_rewards: Rewards from DQN control
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rewards over time
    ax1 = axes[0, 0]
    ax1.plot(fixed_rewards, label='Fixed-Time Control', alpha=0.7, linewidth=2, color='blue')
    ax1.plot(dqn_rewards, label='DQN Control', alpha=0.7, linewidth=2, color='red')
    ax1.set_xlabel('Action Step')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Comparison Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rewards
    ax2 = axes[0, 1]
    fixed_cumulative = np.cumsum(fixed_rewards)
    dqn_cumulative = np.cumsum(dqn_rewards)
    ax2.plot(fixed_cumulative, label='Fixed-Time Control', alpha=0.7, linewidth=2, color='blue')
    ax2.plot(dqn_cumulative, label='DQN Control', alpha=0.7, linewidth=2, color='red')
    ax2.set_xlabel('Action Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Reward Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward distribution
    ax3 = axes[1, 0]
    ax3.hist(fixed_rewards, bins=30, alpha=0.6, label='Fixed-Time Control', color='blue', edgecolor='black')
    ax3.hist(dqn_rewards, bins=30, alpha=0.6, label='DQN Control', color='red', edgecolor='black')
    ax3.set_xlabel('Reward Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    ax4 = axes[1, 1]
    stats_labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
    fixed_stats = [
        np.mean(fixed_rewards),
        np.median(fixed_rewards),
        np.std(fixed_rewards),
        np.min(fixed_rewards),
        np.max(fixed_rewards)
    ]
    dqn_stats = [
        np.mean(dqn_rewards),
        np.median(dqn_rewards),
        np.std(dqn_rewards),
        np.min(dqn_rewards),
        np.max(dqn_rewards)
    ]
    
    x = np.arange(len(stats_labels))
    width = 0.35
    ax4.bar(x - width/2, fixed_stats, width, label='Fixed-Time Control', color='blue', alpha=0.7)
    ax4.bar(x + width/2, dqn_stats, width, label='DQN Control', color='red', alpha=0.7)
    ax4.set_xlabel('Statistic')
    ax4.set_ylabel('Value')
    ax4.set_title('Statistical Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stats_labels, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def main():
    """Compare fixed-time control vs DQN control."""
    
    print("=" * 70)
    print("TRAFFIC LIGHT CONTROL COMPARISON")
    print("=" * 70)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run Fixed-Time Control
    print("\n[1/2] Running Fixed-Time Control...")
    env_fixed = SumoEnvironment(use_gui=False)
    fixed_rewards, _ = run_fixed_time_control(env_fixed, max_steps=100)
    env_fixed.close()
    print(f"   Total reward: {sum(fixed_rewards):.2f}")
    print(f"   Average reward: {np.mean(fixed_rewards):.2f}")
    
    # Run DQN Control (if model exists)
    print("\n[2/2] Running DQN Control...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        env_dqn = SumoEnvironment(use_gui=False)
        state = env_dqn.reset()
        state_size = env_dqn.get_state_size()
        action_size = env_dqn.get_action_size()
        
        agent = DQNAgent(state_size, action_size)
        agent.load(config.MODEL_SAVE_PATH)
        
        dqn_rewards, _ = run_dqn_control(env_dqn, agent, max_steps=100)
        env_dqn.close()
        print(f"   Total reward: {sum(dqn_rewards):.2f}")
        print(f"   Average reward: {np.mean(dqn_rewards):.2f}")
    else:
        print("   DQN model not found. Running random policy instead...")
        env_dqn = SumoEnvironment(use_gui=False)
        state = env_dqn.reset()
        state_size = env_dqn.get_state_size()
        action_size = env_dqn.get_action_size()
        
        agent = DQNAgent(state_size, action_size)
        agent.epsilon = 1.0  # Fully random
        
        dqn_rewards, _ = run_dqn_control(env_dqn, agent, max_steps=100)
        env_dqn.close()
        print(f"   Total reward: {sum(dqn_rewards):.2f}")
        print(f"   Average reward: {np.mean(dqn_rewards):.2f}")
        print("   Note: This is random policy, not trained DQN.")
    
    # Generate comparison plot
    print("\n[3/3] Generating comparison plots...")
    plot_comparison(fixed_rewards, dqn_rewards, "results/control_comparison.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Fixed-Time Control:")
    print(f"  Total Reward:   {sum(fixed_rewards):.2f}")
    print(f"  Average Reward: {np.mean(fixed_rewards):.2f}")
    print(f"  Std Dev:        {np.std(fixed_rewards):.2f}")
    print(f"\nDQN Control:")
    print(f"  Total Reward:   {sum(dqn_rewards):.2f}")
    print(f"  Average Reward: {np.mean(dqn_rewards):.2f}")
    print(f"  Std Dev:        {np.std(dqn_rewards):.2f}")
    
    improvement = ((sum(dqn_rewards) - sum(fixed_rewards)) / abs(sum(fixed_rewards))) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
