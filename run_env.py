import argparse
from environment.custom_env import StorageEnv
import time
import pygame
import sys
import random
import numpy as np

def get_intelligent_random_action(env):
    """
    Get a semi-intelligent random action based on current episode phase
    """
    current_zone = env.grid_states[tuple(env.current_pos)]
    
    # Debug: Print current state
    print(f" {env.episode_phase}, Zone: {current_zone}, Position: {env.current_pos}")
    
    # Phase-based intelligent action selection
    if env.episode_phase == "ANALYZE":
        # In analyze phase, should read conditions
        if not env.has_read_conditions:
            print(f" ANALYZE phase: Reading conditions (action 21)")
            return 21  # Read conditions
        else:
            # Already read, this shouldn't happen but just move to navigate
            print(f" ANALYZE phase: Conditions already read, random movement")
            return random.choice([17, 18, 19, 20])
    
    elif env.episode_phase == "NAVIGATE":
        # In navigate phase, should move towards target zone
        if env.has_read_conditions:
            # Find target zone position
            target_positions = np.where(env.grid_states == env.target_zone)
            if len(target_positions[0]) > 0:
                # Choose closest target position
                target_pos = [target_positions[0][0], target_positions[1][0]]
                current_pos = env.current_pos
                
                # Simple navigation towards target
                if current_pos[0] < target_pos[0]:
                    action = 18  # Move down
                    direction = "down"
                elif current_pos[0] > target_pos[0]:
                    action = 17  # Move up
                    direction = "up"
                elif current_pos[1] < target_pos[1]:
                    action = 20  # Move right
                    direction = "right"
                elif current_pos[1] > target_pos[1]:
                    action = 19  # Move left
                    direction = "left"
                else:
                    # Already at target position
                    action = random.choice([17, 18, 19, 20])
                    direction = "random"
                
                target_zone_name = env.STATE_TYPES[env.target_zone]["name"]
                print(f" NAVIGATE phase: Moving {direction} towards {target_zone_name} (action {action})")
                return action
        
        # Fallback to random movement
        action = random.choice([17, 18, 19, 20])
        action_names = {17: "up", 18: "down", 19: "left", 20: "right"}
        print(f" NAVIGATE phase: Random movement {action_names[action]} (action {action})")
        return action
    
    elif env.episode_phase == "TREAT":
        # In treat phase, should apply appropriate treatment
        if current_zone != -1:
            zone_info = env.STATE_TYPES[current_zone]
            available_actions = [a for a in zone_info["actions"] if a <= 16]  # Only treatment actions
            
            if available_actions:
                action = random.choice(available_actions)
                print(f"TREAT phase: Applying {env.ACTIONS[action]} (action {action})")
                return action
        
        # Fallback - this shouldn't happen in a well-designed episode
        print(f" TREAT phase: No valid treatments, random action")
        return env.action_space.sample()
    
    # Fallback for any unexpected state
    print(f" Unexpected state, random action")
    return env.action_space.sample()

def test_action_space(env):
    """Test that all actions are being generated correctly"""
    print("\n=== Testing Action Space ===")
    print(f"Total actions available: {env.action_space.n}")
    print("Action mappings:")
    for i, action_name in enumerate(env.ACTIONS):
        print(f"  {i:2d}: {action_name}")
    
    print("\nZone-specific actions:")
    for zone_idx, zone_info in enumerate(env.STATE_TYPES):
        print(f"  Zone {zone_idx} ({zone_info['name']}):")
        for action_idx in zone_info['actions']:
            if action_idx < len(env.ACTIONS):
                print(f"    {action_idx}: {env.ACTIONS[action_idx]}")
    print()

def run_continuous_environment(render_mode: str = 'human', intelligent_actions: bool = True, debug_mode: bool = False):
    """
    Complete runner with phase-based episode debugging
    """
    try:
        # Initialize environment
        config = {
            'max_steps': 15,
            'grid_size': (5, 5),
            'obs_type': 'multi'
        }
        env = StorageEnv(config=config, render_mode=render_mode)
        
        print("\n" + "="*60)
        print("Farm Storage Optimization Simulation" if debug_mode else "Farm Storage Optimization Simulation")
        print("="*60)
        print(f"Grid Size: {env.grid_size[0]}x{env.grid_size[1]}")
        print(f"Available Actions: {len(env.ACTIONS)} actions")
        print(f"Rendering Mode: {render_mode}")
        print(f"Action Selection: {'Intelligent Random' if intelligent_actions else 'Pure Random'}")
        print(f"Debug Mode: {debug_mode}")
        print(f"Max Steps per Episode: {config['max_steps']}")
        print("Press Ctrl+C to stop\n")

        if debug_mode:
            test_action_space(env)

        # Print grid layout for reference
        print("Grid Layout:")
        for i in range(env.grid_size[0]):
            row_str = ""
            for j in range(env.grid_size[1]):
                zone_type = env.grid_states[i, j]
                if zone_type == -1:
                    row_str += "  .  "
                else:
                    zone_name = env.STATE_TYPES[zone_type]["name"]
                    row_str += f" {zone_name[:3]} "
            print(f"Row {i}: {row_str}")
        print()

        episode = 0
        action_counts = {}  # Track action frequency
        phase_success_stats = {"ANALYZE": 0, "NAVIGATE": 0, "TREAT": 0}
        total_episodes = 0
        successful_episodes = 0
        
        while True:  # Infinite loop
            episode += 1
            total_episodes += 1
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            step = 0
            episode_actions = []  # Track actions in this episode
            phase_transitions = []  # Track phase changes
            
            print(f"\n=== Episode {episode} ===")
            print(f"Starting Position: {env.current_pos}")
            print(f"Initial Phase: {env.episode_phase}")
            print(f"Environmental Conditions:")
            print(f"  Temperature: {env.temp:.1f}°C")
            print(f"  Humidity: {env.humidity:.1f}%") 
            print(f"  Pest Level: {env.pest_level:.1%}")
            print(f"Target Zone: {env.STATE_TYPES[env.target_zone]['name']}")
            
            current_zone = env.grid_states[tuple(env.current_pos)]
            if current_zone != -1:
                print(f"Starting Zone: {env.STATE_TYPES[current_zone]['name']}")
            else:
                print("Starting Zone: Empty")
            
            last_phase = env.episode_phase
            
            while not (terminated or truncated):
                # Handle PyGame events
                if render_mode == 'human':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                
                # Track phase transitions
                if env.episode_phase != last_phase:
                    phase_transitions.append(f"{last_phase} -> {env.episode_phase}")
                    last_phase = env.episode_phase
                
                # Action selection with debugging
                if debug_mode:
                    print(f"\n  Step {step + 1} - Phase: {env.episode_phase}")
                
                if intelligent_actions:
                    action = get_intelligent_random_action(env)
                else:
                    action = env.action_space.sample()
                    if debug_mode:
                        print(f"    [DEBUG] Pure random action: {action} ({env.ACTIONS[action]})")
                
                # Track action frequency
                action_counts[action] = action_counts.get(action, 0) + 1
                episode_actions.append(action)
                
                # Store previous state for comparison
                prev_pos = env.current_pos.copy()
                prev_zone = env.grid_states[tuple(prev_pos)]
                prev_phase = env.episode_phase
                prev_pest = env.pest_level
                
                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Debug action effects
                if debug_mode:
                    new_pos = env.current_pos
                    new_zone = env.grid_states[tuple(new_pos)]
                    
                    print(f" Action executed: {env.ACTIONS[action]}")
                    print(f" Position change: {prev_pos} -> {new_pos}")
                    print(f" Zone change: {prev_zone} -> {new_zone}")
                    print(f" Phase change: {prev_phase} -> {env.episode_phase}")
                    print(f" Reward: {reward:.2f}")
                    if 'message' in info:
                        print(f" Info: {info['message']}")
                
                # Render
                if render_mode == 'human':
                    env.render()
                    # Update window caption with phase info
                    current_zone = env.grid_states[tuple(env.current_pos)]
                    zone_name = "Empty" if current_zone == -1 else env.STATE_TYPES[current_zone]["name"]
                    pygame.display.set_caption(
                        f"Farm Storage - Episode {episode} | "
                        f"Step {step} | Phase: {env.episode_phase} | "
                        f"Action: {env.ACTIONS[action]} | "
                        f"Pos: {env.current_pos} | Zone: {zone_name} | "
                        f"Pest: {env.pest_level:.1%} | "
                        f"Reward: {total_reward:.1f}"
                    )
                    # Slower for debugging
                    time.sleep(1.5 if debug_mode else 0.8)
                elif render_mode == 'console':
                    current_zone = env.grid_states[tuple(env.current_pos)]
                    zone_name = "Empty" if current_zone == -1 else env.STATE_TYPES[current_zone]["name"]
                    print(f"Step {step:2d}: Phase {env.episode_phase:8s} | "
                          f"Pos {env.current_pos} | "
                          f"Zone: {zone_name:12s} | "
                          f"Action: {env.ACTIONS[action]:25s} | "
                          f"Pest: {env.pest_level:.1%} | "
                          f"Reward: {reward:+6.2f} | "
                          f"Total: {total_reward:+7.1f}")
                    
                    time.sleep(0.3 if debug_mode else 0.15)
            
            # Episode summary with phase analysis
            current_zone = env.grid_states[tuple(env.current_pos)]
            zone_name = "Empty" if current_zone == -1 else env.STATE_TYPES[current_zone]["name"]
            
            print(f"\n--- Episode {episode} Summary ---")
            print(f"Duration: {step} steps")
            print(f"Final Position: {env.current_pos}")
            print(f"Final Zone: {zone_name}")
            print(f"Final Phase: {env.episode_phase}")
            print(f"Total Reward: {total_reward:.1f}")
            print(f"Average Reward per Step: {total_reward/max(step, 1):.2f}")
            
            # Episode outcome analysis
            episode_success = False
            if 'correct_choice' in info and info['correct_choice']:
                print(" SUCCESS: Correct zone selected and treatment applied!")
                successful_episodes += 1
                episode_success = True
            elif terminated:
                if env.episode_phase == "TREAT":
                    print(" FAILED: Wrong zone selected or invalid treatment")
                elif env.episode_phase == "NAVIGATE":
                    print(" INCOMPLETE: Never reached a treatment zone")
                else:
                    print(" FAILED: Never progressed past analysis phase")
            else:
                print(" TIMEOUT: Episode truncated due to step limit")
            
            # Phase transition analysis
            if phase_transitions:
                print(f"Phase Transitions: {' -> '.join(phase_transitions)}")
            else:
                print("Phase Transitions: None (stayed in initial phase)")
            
            # Track phase success
            if env.has_read_conditions:
                phase_success_stats["ANALYZE"] += 1
            if env.chosen_zone is not None:
                phase_success_stats["NAVIGATE"] += 1
            if episode_success:
                phase_success_stats["TREAT"] += 1
            
            # Action diversity analysis
            unique_actions = len(set(episode_actions))
            print(f"Action Diversity: {unique_actions}/{len(env.ACTIONS)} different actions used")
            
            if debug_mode and episode_actions:
                print("Actions taken this episode:")
                action_episode_counts = {}
                for a in episode_actions:
                    action_episode_counts[a] = action_episode_counts.get(a, 0) + 1
                
                for action_id, count in sorted(action_episode_counts.items()):
                    print(f"  {action_id:2d}: {env.ACTIONS[action_id]:25s} x{count}")
            
            # Show target vs chosen zone analysis
            if env.chosen_zone is not None:
                target_name = env.STATE_TYPES[env.target_zone]['name']
                chosen_name = env.STATE_TYPES[env.chosen_zone]['name']
                print(f"Zone Analysis: Target={target_name}, Chosen={chosen_name}")
                if env.chosen_zone == env.target_zone:
                    print(" Correct zone selection!")
                else:
                    print(" Wrong zone selection")
            
            # Overall statistics every 10 episodes
            if episode % 10 == 0:
                success_rate = (successful_episodes / total_episodes) * 100
                print(f"\n=== Overall Statistics (Episodes 1-{episode}) ===")
                print(f"Success Rate: {successful_episodes}/{total_episodes} ({success_rate:.1f}%)")
                print(f"Phase Success Rates:")
                print(f"  ANALYZE: {phase_success_stats['ANALYZE']}/{total_episodes} ({(phase_success_stats['ANALYZE']/total_episodes)*100:.1f}%)")
                print(f"  NAVIGATE: {phase_success_stats['NAVIGATE']}/{total_episodes} ({(phase_success_stats['NAVIGATE']/total_episodes)*100:.1f}%)")
                print(f"  TREAT: {phase_success_stats['TREAT']}/{total_episodes} ({(phase_success_stats['TREAT']/total_episodes)*100:.1f}%)")
                
                # Show most common actions
                if action_counts:
                    total_actions = sum(action_counts.values())
                    print(f"Most Common Actions:")
                    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
                    for action_id, count in sorted_actions[:5]:
                        percentage = (count / total_actions) * 100
                        print(f"  {action_id:2d}: {env.ACTIONS[action_id]:25s} - {count:3d} times ({percentage:5.1f}%)")
            
            if render_mode == 'human':
                time.sleep(3.0 if debug_mode else 2.0)  # Longer pause for debugging
            else:
                time.sleep(1.0 if debug_mode else 0.5)
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        
        # Final comprehensive report
        if total_episodes > 0:
            success_rate = (successful_episodes / total_episodes) * 100
            print(f"\n=== Final Report ===")
            print(f"Total Episodes: {total_episodes}")
            print(f"Successful Episodes: {successful_episodes} ({success_rate:.1f}%)")
            print(f"Phase Success Rates:")
            print(f"  ANALYZE (read conditions): {phase_success_stats['ANALYZE']}/{total_episodes} ({(phase_success_stats['ANALYZE']/total_episodes)*100:.1f}%)")
            print(f"  NAVIGATE (reach zone): {phase_success_stats['NAVIGATE']}/{total_episodes} ({(phase_success_stats['NAVIGATE']/total_episodes)*100:.1f}%)")
            print(f"  TREAT (correct treatment): {phase_success_stats['TREAT']}/{total_episodes} ({(phase_success_stats['TREAT']/total_episodes)*100:.1f}%)")
        
        # Final action frequency report
        if action_counts:
            print(f"\nFinal Action Frequency Report:")
            total_actions = sum(action_counts.values())
            print(f"Total actions taken: {total_actions}")
            
            print("\nAll Actions:")
            for action_id in sorted(action_counts.keys()):
                count = action_counts[action_id]
                percentage = (count / total_actions) * 100
                print(f"  {action_id:2d}: {env.ACTIONS[action_id]:25s} - {count:3d} times ({percentage:5.1f}%)")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        if render_mode == 'human':
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
    parser.add_argument('--random', action='store_true',
                      help='Use pure random actions instead of intelligent random')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with detailed logging')
    
    args = parser.parse_args()
    
    print("Starting Farm Storage Optimization Environment...")
    print(f"Mode: {'Pure Random' if args.random else 'Intelligent Random'}")
    print(f"Render: {args.render}")
    print(f"Debug: {args.debug}")
    
    run_continuous_environment(
        render_mode=args.render, 
        intelligent_actions=not args.random,
        debug_mode=args.debug
    )