import argparse
from environment.custom_env import StorageEnv
import time
import pygame
import sys

def run_continuous_environment(render_mode: str = 'human', max_episodes: int = None):
    """
    Run the environment continuously with clean visualization.
    """
    try:
        # Initialize environment
        env = StorageEnv(render_mode=render_mode)
        print("Environment initialized successfully")

        episode = 0
        
        while max_episodes is None or episode < max_episodes:
            episode += 1
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            step = 0
            
            print(f"\n=== Episode {episode} ===")
            
            while not terminated:
                # Handle PyGame events
                if render_mode == 'human':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                
                # Action selection (replace with your agent)
                action = env.action_space.sample()
                
                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                if render_mode == 'human':
                    env.render()
                    pygame.display.set_caption(
                        f"Farm Storage - Day {env.day} | "
                        f"Pest: {env.pest_level:.1%} | "
                        f"Temp: {env.state['temp'][0]:.1f}°C"
                    )
                elif render_mode == 'console':
                    env.render()
                    time.sleep(0.1)  # Small delay for readability
                
                # Early termination if pests take over
                if env.pest_level >= 0.95:
                    terminated = True
            
            print(f"Completed in {env.day} days")
            print(f"Final pest level: {env.pest_level:.1%}")
            
            # Brief pause between episodes
            if render_mode == 'human':
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        env.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Farm Storage Environment')
    parser.add_argument('--render', type=str, default='human',
                      choices=['human', 'console'], help='Rendering mode')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Max episodes to run (None for infinite)')
    
    args = parser.parse_args()
    
    print("Farm Storage Optimization Simulation")
    print("-----------------------------------")
    print(f"Rendering mode: {args.render}")
    print("Press Ctrl+C to stop\n")
    
    run_continuous_environment(
        render_mode=args.render,
        max_episodes=args.episodes
    )