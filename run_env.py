import argparse
from environment.custom_env import StorageEnv
import time
import pygame
import sys

def run_continuous_environment(render_mode: str = 'human', max_episodes: int = None):
    """
    Enhanced runner for the Farm Storage Optimization Environment
    with full support for spatial grid system and recommendations
    """
    try:
        # Initialize environment
        env = StorageEnv(render_mode=render_mode)
        print("\n" + "="*50)
        print("Farm Storage Optimization Simulation")
        print("="*50)
        print(f"Grid Size: {env.grid_size[0]}x{env.grid_size[1]}")
        print(f"Available Actions: {len(env.ACTIONS)} actions")
        print(f"Rendering Mode: {render_mode}")
        print("Press Ctrl+C to stop\n")

        episode = 0
        
        while max_episodes is None or episode < max_episodes:
            episode += 1
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            step = 0
            
            print(f"\n=== Episode {episode} ===")
            print(f"Starting Position: {env.current_pos}")
            print(f"Initial Pest Level: {env.pest_level:.1%}")
            print(f"Initial Weather: {env.state['temp'][0]:.1f}°C, "
                  f"{env.state['humidity'][0]:.1f}% humidity")
            
            while not terminated:
                # Handle PyGame events
                if render_mode == 'human':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                
                # Action selection (replace this with your agent)
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
                        f"Pos: {env.current_pos} | "
                        f"Pest: {env.pest_level:.1%} | "
                        f"Temp: {env.state['temp'][0]:.1f}°C | "
                        f"Reward: {total_reward:.1f}"
                    )
                elif render_mode == 'console':
                    if step % 5 == 0 or terminated:  # Reduce console spam
                        print(f"Day {env.day}: Pos {env.current_pos} | "
                              f"Action: {env.ACTIONS[action]} | "
                              f"Pest: {env.pest_level:.1%} | "
                              f"Reward: {total_reward:.1f}")
                        if 'recommended' in info:
                            print(f"Recommended: {info['recommended']}")
                    time.sleep(0.1)
                
                # Early termination if pests take over
                if env.pest_level >= 0.95:
                    terminated = True
            
            # Episode summary
            print(f"\nEpisode {episode} completed in {env.day} days")
            print(f"Final Position: {env.current_pos}")
            print(f"Final Pest Level: {env.pest_level:.1%}")
            print(f"Total Reward: {total_reward:.1f}")
            print(f"Last Action: {info['action']}")
            print(f"Recommended Action: {info['recommended']}")
            
            # Pause between episodes
            if render_mode == 'human':
                time.sleep(1.5)  # Longer pause to observe results
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        env.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Farm Storage Optimization Environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--render', type=str, default='human',
                      choices=['human', 'console'], 
                      help='Rendering mode (human=visual, console=text)')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Maximum number of episodes to run (None for infinite)')
    
    args = parser.parse_args()
    
    run_continuous_environment(
        render_mode=args.render,
        max_episodes=args.episodes
    )