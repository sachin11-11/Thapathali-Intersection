"""
Quick test script to verify the SUMO environment and DQN agent work.
"""
from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent

def test_environment():
    print("Testing SUMO Environment...")
    print("=" * 70)
    
    # Initialize environment
    env = SumoEnvironment(use_gui=False)
    
    # Reset and get initial state
    state = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  State size: {env.get_state_size()}")
    print(f"  Action size: {env.get_action_size()}")
    print(f"  Incoming lanes: {len(env.incoming_lanes)}")
    print(f"  Outgoing lanes: {len(env.outgoing_lanes)}")
    print(f"  Initial state shape: {state.shape}")
    
    # Test a few random actions
    print("\nTesting actions...")
    for action in range(min(4, env.get_action_size())):
        next_state, reward, done, info = env.step(action)
        print(f"  Action {action}: Reward = {reward:.2f}, Done = {done}, Steps = {info['step_count']}")
        if done:
            print("  Simulation ended early")
            break
    
    env.close()
    print("\n✓ Environment test passed!")
    return env.get_state_size(), env.get_action_size()


def test_agent(state_size, action_size):
    print("\n" + "=" * 70)
    print("Testing DQN Agent...")
    print("=" * 70)
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    print(f"✓ Agent initialized")
    print(f"  Device: {agent.device}")
    print(f"  Q-Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Test action selection
    import numpy as np
    dummy_state = np.zeros(state_size, dtype=np.float32)
    action = agent.act(dummy_state, training=False)
    print(f"✓ Action selection works: Selected action {action}")
    
    # Test experience replay
    print(f"\nTesting experience replay...")
    for i in range(100):
        state = np.random.rand(state_size).astype(np.float32)
        action = np.random.randint(0, action_size)
        reward = np.random.randn()
        next_state = np.random.rand(state_size).astype(np.float32)
        done = False
        agent.remember(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {agent.replay_buffer.size()}")
    loss = agent.replay()
    print(f"  Training loss: {loss:.4f}")
    print(f"✓ Experience replay works!")
    
    print("\n✓ Agent test passed!")


if __name__ == "__main__":
    try:
        state_size, action_size = test_environment()
        test_agent(state_size, action_size)
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nYou can now run: python3 train_dqn.py")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
