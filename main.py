"""
A Play Script for StorageEnv, Visualizing the trained agents in a new environment.
Supports A2C, DQN, PPO, and REINFORCE algorithms
"""

import os
import sys
import time
import argparse
import numpy as np

# Adding project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import StorageEnv

def get_algorithm_class(algorithm_name):
    """Get the appropriate algorithm class based on name"""
    algorithms = {
        'a2c': A2C,
        'dqn': DQN, 
        'ppo': PPO,
    }
    
    # Handle REINFORCE specially
    if algorithm_name.lower() == 'reinforce':
        if REINFORCE_AVAILABLE and REINFORCE_CLASS:
            return REINFORCE_CLASS
        else:
            print(" REINFORCE not available. Trying to load as PPO...")
            return PPO  # Fallback to PPO
    
    return algorithms.get(algorithm_name.lower())

def play_model(model_path, algorithm='a2c', obs_type='mlp', episodes=3, 
               curriculum=2, speed=1.0, deterministic=True, pause_on_action=False):
    """
    Play/watch a trained model interact with the StorageEnv
    
    Args:
        model_path (str): Path to the trained model
        algorithm (str): Algorithm type ('a2c', 'dqn', 'ppo', 'reinforce')  
        obs_type (str): Observation type ('mlp', 'cnn', 'multi')
        episodes (int): Number of episodes to play
        curriculum (int): Difficulty level (1-3)
        speed (float): Playback speed multiplier (higher = faster)
        deterministic (bool): Use deterministic policy
        pause_on_action (bool): Pause after each action for manual control
    """
    
    algorithm_class = get_algorithm_class(algorithm)
    if algorithm_class is None:
        print(f" Unsupported algorithm: {algorithm}")
        print(f" Supported algorithms: a2c, dqn, ppo, reinforce")
        return None
    
    # Create environment with console rendering
    env_config = {
        'max_steps': 25,
        'initial_pest': 0.1,
        'location': "Kigali", 
        'grid_size': (5, 5),
        'layout': 'custom',
        'obs_type': obs_type,
        'curriculum_stage': curriculum,
        'use_action_masking': False,
        'intermediate_rewards': True
    }
    
    env = StorageEnv(config=env_config, render_mode='human')
    env = DummyVecEnv([lambda: env])
    
    # Load model with error handling for different formats
    try:
        model = algorithm_class.load(model_path)
        print(f" Loaded {algorithm.upper()} model from {os.path.basename(model_path)}")
    except Exception as e:
        print(f" Error loading {algorithm.upper()} model: {e}")
        
        # Try alternative loading methods
        if algorithm.lower() == 'reinforce':
            print(" Trying alternative REINFORCE loading methods...")
            try:
                # Try loading as PPO if REINFORCE fails
                model = PPO.load(model_path)
                print(f" Loaded model as PPO (REINFORCE fallback)")
                algorithm = 'ppo'
            except Exception as e2:
                print(f" Alternative loading failed: {e2}")
                return None
        else:
            return None
    
    print(f"\n Starting playback...")
    print(f" Config: {obs_type.upper()} obs, Curriculum Stage {curriculum}, Speed {speed}x")
    print(f" Deterministic: {'ON' if deterministic else 'OFF'}")
    print(f" Pause mode: {'ON' if pause_on_action else 'OFF'}")
    print("=" * 80)
    
    episode_stats = []
    
    try:
        for episode in range(episodes):
            print(f"\n Episode {episode + 1}/{episodes}")
            print("-" * 50)
            
            obs = env.reset()
            done = [False]
            truncated = [False]
            total_reward = 0
            step_count = 0
            phase_history = []
            correct_choice = False
            target_zone_name = ""
            chosen_zone_name = ""
            conditions_read = False
            navigation_started = False
            zone_reached = False
            
            # Show initial state
            env.render()
            if pause_on_action:
                input(" Press Enter to start episode...")
            else:
                time.sleep(2.0 / speed)
            
            while not (done[0] or truncated[0]):
                # Get action from model with error handling
                try:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    action_val = action[0] if isinstance(action, (list, np.ndarray)) else action
                except Exception as e:
                    print(f" Prediction error: {e}")
                    # Use random action as fallback
                    action_val = env.action_space.sample()
                    print(f" Using random action as fallback")
                
                # Ensure action is valid
                if hasattr(env.envs[0], 'ACTIONS') and action_val < len(env.envs[0].ACTIONS):
                    action_name = env.envs[0].ACTIONS[action_val]
                else:
                    action_name = f"Action_{action_val}"
                
                print(f"\n Agent chooses: {action_name} (Action {action_val})")
                
                # Take action
                obs, reward, done, info = env.step([action_val])
                
                # Handle vectorized environment returns
                reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                done_val = done[0] if isinstance(done, (list, np.ndarray)) else done
                truncated_val = truncated[0] if isinstance(truncated, (list, np.ndarray)) else False
                info_dict = info[0] if isinstance(info, list) else info
                
                total_reward += reward_val
                step_count += 1
                
                # Track episode progress
                current_phase = info_dict.get('phase', 'UNKNOWN')
                if current_phase not in phase_history:
                    phase_history.append(current_phase)
                
                # Update progress flags
                if not conditions_read and current_phase == 'NAVIGATE':
                    conditions_read = True
                    print(" Conditions analyzed successfully!")
                
                if not navigation_started and current_phase == 'NAVIGATE':
                    navigation_started = True
                
                if not zone_reached and current_phase == 'TREAT':
                    zone_reached = True
                    print(" Zone reached!")
                
                # Show updated state
                env.render()
                
                # Display step results
                print(f" Reward: {reward_val:+.2f} | Total: {total_reward:.2f}")
                
                if 'message' in info_dict:
                    print(f" Info: {info_dict['message']}")
                
                # Track final results
                if 'target_zone_name' in info_dict:
                    target_zone_name = info_dict['target_zone_name']
                if 'chosen_zone_name' in info_dict:
                    chosen_zone_name = info_dict['chosen_zone_name']
                if 'correct_choice' in info_dict:
                    correct_choice = info_dict['correct_choice']
                
                # Control playback speed
                if pause_on_action:
                    user_input = input(" Press Enter to continue (or 'q' to quit, 's' to skip episode): ")
                    if user_input.lower() == 'q':
                        print(" Stopping playback...")
                        env.close()
                        return episode_stats
                    elif user_input.lower() == 's':
                        print(" Skipping to next episode...")
                        break
                else:
                    time.sleep(1.5 / speed)
                
                # Safety break
                if step_count > env_config['max_steps'] + 5:
                    print(" Episode too long, ending...")
                    break
            
            
            print(f"\n Episode {episode + 1} Complete:")
            print(f" Target Zone: {target_zone_name}")
            print(f" Chosen Zone: {chosen_zone_name}")
            print(f" Correct Choice: {'YES' if correct_choice else 'NO'}")
            print(f" Total Reward: {total_reward:.2f}")
            print(f" Steps Taken: {step_count}")
            print(f" Phases: {' → '.join(phase_history)}")
            
            # Progress analysis
            progress_items = []
            if conditions_read:
                progress_items.append(" Conditions Read")
            if navigation_started:
                progress_items.append(" Navigation Started")
            if zone_reached:
                progress_items.append(" Zone Reached")
            if correct_choice:
                progress_items.append(" Correct Treatment")
            
            print(f" Progress: {', '.join(progress_items) if progress_items else 'None'}")
            
            # Performance rating
            if correct_choice and step_count <= 15:
                rating = " EXCELLENT"
            elif correct_choice and step_count <= 20:
                rating = " GOOD"
            elif correct_choice:
                rating = " OK (slow)"
            elif zone_reached:
                rating = "WRONG CHOICE"
            elif navigation_started:
                rating = "NAVIGATION FAILED"
            else:
                rating = " ANALYSIS FAILED"
            
            print(f"Rating: {rating}")
            
            # Store episode stats
            episode_stats.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': step_count,
                'success': correct_choice,
                'phases_completed': len(phase_history),
                'conditions_read': conditions_read,
                'navigation_started': navigation_started,
                'zone_reached': zone_reached,
                'target_zone': target_zone_name,
                'chosen_zone': chosen_zone_name,
                'rating': rating,
                'algorithm': algorithm.upper()
            })
            
            if episode < episodes - 1:  # Not last episode
                if pause_on_action:
                    input("\n Press Enter for next episode...")
                else:
                    print(" Next episode in 3 seconds...")
                    time.sleep(3.0 / speed)
    
    except KeyboardInterrupt:
        print("\n Playback interrupted by user")
    
    finally:
        env.close()
        
        # Final summary
        if episode_stats:
            print(f"\n FINAL SUMMARY ({len(episode_stats)} episodes)")
            print("=" * 50)
            
            total_episodes = len(episode_stats)
            successful_episodes = sum(1 for ep in episode_stats if ep['success'])
            avg_reward = np.mean([ep['total_reward'] for ep in episode_stats])
            avg_steps = np.mean([ep['steps'] for ep in episode_stats])
            
            print(f" Algorithm: {algorithm.upper()}")
            print(f" Success Rate: {successful_episodes}/{total_episodes} ({successful_episodes/total_episodes*100:.1f}%)")
            print(f" Average Reward: {avg_reward:.2f}")
            print(f" Average Steps: {avg_steps:.1f}")
            
            # Phase completion stats
            conditions_read_count = sum(1 for ep in episode_stats if ep['conditions_read'])
            navigation_count = sum(1 for ep in episode_stats if ep['navigation_started'])
            zone_reached_count = sum(1 for ep in episode_stats if ep['zone_reached'])
            
            print(f" Conditions Read: {conditions_read_count}/{total_episodes} ({conditions_read_count/total_episodes*100:.1f}%)")
            print(f" Navigation Started: {navigation_count}/{total_episodes} ({navigation_count/total_episodes*100:.1f}%)")
            print(f" Zones Reached: {zone_reached_count}/{total_episodes} ({zone_reached_count/total_episodes*100:.1f}%)")
            
            # Overall model assessment
            if successful_episodes / total_episodes >= 0.8:
                assessment = " EXCELLENT - Model performs very well!"
            elif successful_episodes / total_episodes >= 0.6:
                assessment = " GOOD - Model is competent with room for improvement"
            elif successful_episodes / total_episodes >= 0.3:
                assessment = " MODERATE - Model needs more training"
            else:
                assessment = " POOR - Model requires significant improvement"
            
            print(f"\n Overall Assessment: {assessment}")
        
        print("\n Playback finished!")
        return episode_stats

def compare_models(model_paths, episodes=5, curriculum=2, obs_type='mlp', speed=1.0):
    """
    Compare multiple models side by side
    
    Args:
        model_paths (dict): Dictionary of {algorithm_name: model_path}
        episodes (int): Number of episodes per model
        curriculum (int): Difficulty level
        obs_type (str): Observation type
        speed (float): Playback speed
    """
    print(f" COMPARING {len(model_paths)} MODELS")
    print("=" * 60)
    
    all_stats = {}
    
    for algorithm, model_path in model_paths.items():
        print(f"\n Testing {algorithm.upper()} model...")
        print("-" * 40)
        
        stats = play_model(
            model_path=model_path,
            algorithm=algorithm,
            obs_type=obs_type,
            episodes=episodes,
            curriculum=curriculum,
            speed=speed,
            deterministic=True,
            pause_on_action=False
        )
        
        if stats:
            all_stats[algorithm] = stats
    
    # Comparison summary
    if len(all_stats) > 1:
        print(f"\n MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        comparison_data = []
        for algorithm, stats in all_stats.items():
            success_rate = sum(1 for ep in stats if ep['success']) / len(stats)
            avg_reward = np.mean([ep['total_reward'] for ep in stats])
            avg_steps = np.mean([ep['steps'] for ep in stats])
            
            comparison_data.append({
                'algorithm': algorithm.upper(),
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps
            })
        
        # Sort by success rate
        comparison_data.sort(key=lambda x: x['success_rate'], reverse=True)
        
        print(f"{'Rank':<4} {'Algorithm':<10} {'Success %':<10} {'Avg Reward':<12} {'Avg Steps':<10}")
        print("-" * 50)
    
        
        # Best performing model
        best_model = comparison_data[0]
        print(f"\n Best Performing Model: {best_model['algorithm']}")
        print(f"   Success Rate: {best_model['success_rate']*100:.1f}%")
        print(f"   Average Reward: {best_model['avg_reward']:.2f}")
        print(f"   Average Steps: {best_model['avg_steps']:.1f}")

def main():
    """Command line interface for play script"""
    parser = argparse.ArgumentParser(description="Play/watch trained StorageEnv models")
    
    parser.add_argument("model_path", help="Path to the trained model (.zip file)")
    parser.add_argument("-a", "--algorithm", default="a2c", 
                       choices=['a2c', 'dqn', 'ppo', 'reinforce'],
                       help="Algorithm type (default: a2c)")
    parser.add_argument("-o", "--obs-type", default="mlp", choices=['mlp', 'cnn', 'multi'],
                       help="Observation type (default: mlp)")
    parser.add_argument("-e", "--episodes", type=int, default=3,
                       help="Number of episodes to play (default: 3)")
    parser.add_argument("-c", "--curriculum", type=int, default=2, choices=[1,2,3],
                       help="Curriculum stage 1=Easy, 2=Medium, 3=Full (default: 2)")
    parser.add_argument("-s", "--speed", type=float, default=1.0,
                       help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--random", action="store_true",
                       help="Use stochastic policy instead of deterministic")
    parser.add_argument("--pause", action="store_true", 
                       help="Pause after each action for manual stepping")
    parser.add_argument("--compare", nargs="*", metavar="MODEL_PATH",
                       help="Compare multiple models (provide additional model paths)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f" Model file not found: {args.model_path}")
        return
    
    if not args.model_path.endswith('.zip'):
        print(" Warning: Model path should typically end with .zip")
    
    print(f" StorageEnv Model Player")
    print(f" Model: {os.path.basename(args.model_path)}")
    print(f" Algorithm: {args.algorithm.upper()}")
    print(f" Observation: {args.obs_type.upper()}")
    print(f" Curriculum: Stage {args.curriculum}")
    
    # Handle comparison mode
    if args.compare:
        print(f"\n Comparison mode enabled!")
        model_paths = {args.algorithm: args.model_path}
        
        # Add additional models for comparison
        for i, model_path in enumerate(args.compare):
            if os.path.exists(model_path):
                # Try to infer algorithm from filename
                filename = os.path.basename(model_path).lower()
                if 'ppo' in filename:
                    alg = 'ppo'
                elif 'dqn' in filename:
                    alg = 'dqn'
                elif 'a2c' in filename:
                    alg = 'a2c'
                elif 'reinforce' in filename:
                    alg = 'reinforce'
                else:
                    alg = f'model_{i+2}'  # fallback name
                
                model_paths[alg] = model_path
            else:
                print(f" Comparison model not found: {model_path}")
        
        stats = compare_models(
            model_paths=model_paths,
            episodes=args.episodes,
            curriculum=args.curriculum,
            obs_type=args.obs_type,
            speed=args.speed
        )
    else:
        # Single model mode
        stats = play_model(
            model_path=args.model_path,
            algorithm=args.algorithm,
            obs_type=args.obs_type,
            episodes=args.episodes,
            curriculum=args.curriculum,
            speed=args.speed,
            deterministic=not args.random,
            pause_on_action=args.pause
        )
    
    return stats

if __name__ == "__main__":
    # Command line usage
    main()
    
    # Example usage in code:
    # 
    # # Single model
    # play_model(
    #     model_path="./models/ppo/ppo_storage_model.zip",
    #     algorithm='ppo',
    #     obs_type='mlp', 
    #     episodes=5,
    #     curriculum=2,
    #     speed=1.5
    # )
    #
    # # Compare multiple models
    # compare_models({
    #     'a2c': './models/a2c/a2c_storage_model.zip',
    #     'ppo': './models/ppo/ppo_storage_model.zip',
    #     'dqn': './models/dqn/dqn_storage_model.zip'
    # }, episodes=5)
    # python play.py ./models/a2c/a2c_model.zip --compare ./models/ppo/ppo_model.zip ./models/dqn/dqn_model.zip
