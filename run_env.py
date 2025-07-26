# run_env.py
import pygame
import numpy as np
from environment.custom_env import StorageEnv

def test_environment(episodes: int = 3, steps_per_episode: int = 100, step_delay: int = 500):
    """Test the environment with visualization
    
    Args:
        episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
        step_delay: Milliseconds delay between steps (default 500ms)
    """
    env = None
    try:
        # Initialize environment with visualization
        env = StorageEnv(render_mode='human')
        
        # Verify visualization
        if not hasattr(env, 'visualizer') or env.visualizer is None:
            raise ImportError("Visualization failed - check pygame installation")
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Render first to show initial state
                frame = env.render()
                if frame is None:  # Window closed
                    print("Window closed by user")
                    break
                
                # Get action (random for demo)
                action = env.action_space.sample()
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Print step info - accessing state variables safely
                print(f"\nEpisode {episode+1}, Step {step+1}:")
                print(f"Position: ({env.agent_pos[0]}, {env.agent_pos[1]})")
                print(f"Action: {env.ACTIONS[action]}")
                print(f"Reward: {reward:.2f} (Total: {episode_reward:.2f})")
                print(f"Condition: {info['condition']}")
                print(f"Recommended: {info['recommended']}")
                print(f"Weather: {info['weather'][0]:.1f}°C, {info['weather'][1]:.1f}%")
                
                # Check termination
                if terminated or truncated:
                    print(f"Episode finished after {step+1} steps")
                    break
                
                # Add delay between steps (in milliseconds)
                pygame.time.delay(step_delay)
            
            print(f"\n=== Episode {episode+1} Summary ===")
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Final position: ({env.agent_pos[0]}, {env.agent_pos[1]})")
            print(f"Final risk: {env._get_current_risk()*100:.1f}%")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        if env is not None:
            env.close()
        pygame.quit()

if __name__ == "__main__":
    # Run with error handling
    try:
        # You can adjust the step_delay parameter to control speed
        # Higher values = slower animation (500 = half second between steps)
        test_environment(episodes=10000, steps_per_episode=100, step_delay=500)
    except Exception as e:
        print(f"Fatal error: {str(e)}")