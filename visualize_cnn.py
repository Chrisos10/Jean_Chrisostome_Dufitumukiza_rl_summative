#!/usr/bin/env python3
"""
Play Script for StorageEnv - Watch your trained agent in action!
Similar to OpenAI Gym's play functionality
"""

import time
import argparse
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import StorageEnv

def play_model(model_path, algorithm='a2c', obs_type='mlp', episodes=5, 
               curriculum=1, speed=1.0, deterministic=True, pause_on_action=False):
    """
    Play/watch a trained model interact with the StorageEnv
    
    Args:
        model_path (str): Path to the trained model
        algorithm (str): Algorithm type ('a2c', 'ppo', 'dqn')  
        obs_type (str): Observation type ('mlp', 'cnn', 'multi')
        episodes (int): Number of episodes to play
        curriculum (int): Difficulty level (0-3)
        speed (float): Playback speed multiplier (higher = faster)
        deterministic (bool): Use deterministic policy
        pause_on_action (bool): Pause after each action for manual control
    """
    
    algorithms = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN}
    
    # Create environment with human rendering
    env_config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali", 
        'grid_size': (5, 5),
        'layout': 'custom',
        'obs_type': obs_type,
        'curriculum_level': curriculum
    }
    
    env = StorageEnv(config=env_config, render_mode='human')
    env = DummyVecEnv([lambda: env])
    
    # Load model
    try:
        model = algorithms[algorithm.lower()].load(model_path)
        print(f"Loaded {algorithm.upper()} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"\nStarting playback...")
    print(f"Config: {obs_type.upper()} obs, Curriculum {curriculum}, Speed {speed}x")
    print(f"Pause mode: {'ON' if pause_on_action else 'OFF'}")
    print("=" * 60)
    
    try:
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            print("-" * 40)
            
            obs = env.reset()
            done = [False]
            total_reward = 0
            step_count = 0
            critical_zone_time = 0
            consecutive_critical_steps = 0
            
            # Show initial state
            env.render()
            if pause_on_action:
                input("Press Enter to start episode...")
            else:
                time.sleep(2.0 / speed)
            
            while not done[0]:
                # Get action from model
                action, _states = model.predict(obs, deterministic=deterministic)
                action_val = action[0]
                action_name = env.envs[0].ACTIONS[action_val]
                
                print(f"\nAgent chooses: {action_name} (Action {action_val})")
                
                # Take action
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                step_count += 1
                
                # Show updated state
                env.render()
                
                # Display step results
                step_info = info[0]
                
                # Track critical zone time
                if step_info['zone_type'] == "Critical Risk":
                    critical_zone_time += 1
                    consecutive_critical_steps += 1
                    zone_status = f"CRITICAL ZONE! (Total: {critical_zone_time} steps, Consecutive: {consecutive_critical_steps})"
                else:
                    if consecutive_critical_steps > 0:
                        print(f"Escaped critical zone after {consecutive_critical_steps} consecutive steps!")
                    consecutive_critical_steps = 0
                    zone_status = f"Zone: {step_info['zone_type']}"
                
                print(f"Position: {step_info['position']} | {zone_status}")
                print(f"Pest Level: {step_info['pest_level']:.3f} | Reward: {reward[0]:+.2f}")
                print(f"Total Reward: {total_reward:.2f} | Day: {step_info['day']}")
                
                # Warning for extended critical zone time
                if consecutive_critical_steps >= 3:
                    print(f"WARNING: Agent has been in critical zone for {consecutive_critical_steps} consecutive steps!")
                elif critical_zone_time > 0 and step_info['zone_type'] != "Critical Risk":
                    print(f"Critical zone exposure this episode: {critical_zone_time} total steps")
                
                # Control playback speed
                if pause_on_action:
                    user_input = input("Press Enter to continue (or 'q' to quit, 's' to skip episode): ")
                    if user_input.lower() == 'q':
                        print("Stopping playback...")
                        env.close()
                        return
                    elif user_input.lower() == 's':
                        print("Skipping to next episode...")
                        break
                else:
                    time.sleep(1.5 / speed)
                
                # Safety break
                if step_count > 100:
                    print("Episode too long, ending...")
                    break
            
            # Episode summary
            final_pest = step_info['pest_level']
            success = "SUCCESS" if final_pest < 0.5 else "FAILURE"
            
            print(f"\nEpisode {episode + 1} Complete: {success}")
            print(f"   Final Pest Level: {final_pest:.3f}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Steps Taken: {step_count}")
            print(f"   CRITICAL ZONE Time: {critical_zone_time} steps ({critical_zone_time/step_count*100:.1f}% of episode)")
            
            # Critical zone analysis
            if critical_zone_time == 0:
                print(f"   Perfect! Agent avoided all critical zones")
            elif critical_zone_time <= 2:
                print(f"   Good! Minimal critical zone exposure")
            elif critical_zone_time <= 5:
                print(f"   Moderate critical zone exposure - room for improvement")
            else:
                print(f"   High critical zone exposure - agent needs better navigation")
            
            if episode < episodes - 1:  # Not last episode
                if pause_on_action:
                    input("\nPress Enter for next episode...")
                else:
                    print("Next episode in 3 seconds...")
                    time.sleep(3.0 / speed)
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    
    finally:
        env.close()
        print("Playback finished!")

def main():
    """Command line interface for play script"""
    parser = argparse.ArgumentParser(description="Play/watch trained StorageEnv models")
    
    parser.add_argument("model_path", help="Path to the trained model (.zip file)")
    parser.add_argument("-a", "--algorithm", default="a2c", choices=['a2c', 'ppo', 'dqn'],
                       help="Algorithm type (default: a2c)")
    parser.add_argument("-o", "--obs-type", default="mlp", choices=['mlp', 'cnn', 'multi'],
                       help="Observation type (default: mlp)")
    parser.add_argument("-e", "--episodes", type=int, default=3,
                       help="Number of episodes to play (default: 3)")
    parser.add_argument("-c", "--curriculum", type=int, default=1, choices=[0,1,2,3],
                       help="Curriculum level 0=Easy, 1=Medium, 2=Hard, 3=Expert (default: 1)")
    parser.add_argument("-s", "--speed", type=float, default=1.0,
                       help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--random", action="store_true",
                       help="Use stochastic policy instead of deterministic")
    parser.add_argument("--pause", action="store_true", 
                       help="Pause after each action for manual stepping")
    
    args = parser.parse_args()
    
    # Validate model path
    if not args.model_path.endswith('.zip'):
        print("Model path should end with .zip")
        return
    
    print(f"StorageEnv Model Player")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Observation: {args.obs_type.upper()}")
    
    play_model(
        model_path=args.model_path,
        algorithm=args.algorithm,
        obs_type=args.obs_type,
        episodes=args.episodes,
        curriculum=args.curriculum,
        speed=args.speed,
        deterministic=not args.random,
        pause_on_action=args.pause
    )

if __name__ == "__main__":
    # Command line usage
    main()
    
    # Or use directly in code:
    # play_model(
    #     model_path="./logs/a2c_run2/a2c_storage_model.zip",
    #     algorithm='a2c',
    #     obs_type='mlp', 
    #     episodes=3,
    #     curriculum=1,
    #     speed=1.5,
    #     pause_on_action=False
    # )