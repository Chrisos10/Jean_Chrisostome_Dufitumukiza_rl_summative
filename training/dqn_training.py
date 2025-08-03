import os
import sys
# Adding project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from environment.custom_env import StorageEnv

class DQNMetricsCallback(BaseCallback):
    """ callback class to collect DQN training metrics and save as CSV files"""
    
    def __init__(self, save_freq: int = 1000, save_path: str = "./logs/", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
        # Episode-level metrics
        self.episode_data = []  # Store complete episode records
        self.episode_count = 0
        self.cumulative_reward_sum = 0.0
        
        # Training step metrics
        self.training_data = []  # Store complete training step records
        
        # For moving averages
        self.recent_rewards = []  # Keep track of recent rewards for moving averages
        
        # Track episodes from the rollout buffer
        self.temp_episode_rewards = []
        self.temp_episode_lengths = []
        self.temp_episode_success_metrics = []
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_rollout_start(self) -> None:
        """Called before starting a new rollout"""
        # Clear temporary storage
        self.temp_episode_rewards = []
        self.temp_episode_lengths = []
        self.temp_episode_success_metrics = []
    
    def _on_step(self) -> bool:
        """Called at each environment step - collect episode info here"""
        
        # Check if we have episode completion info
        if 'infos' in self.locals and self.locals['infos'] is not None:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    episode_info = info['episode']
                    episode_reward = float(episode_info.get('r', 0))
                    episode_length = int(episode_info.get('l', 0))
                    
                    # Get success metrics if available
                    success_metrics = info.get('success_metrics', {})
                    
                    # Store temporarily
                    self.temp_episode_rewards.append(episode_reward)
                    self.temp_episode_lengths.append(episode_length)
                    self.temp_episode_success_metrics.append(success_metrics)
        
        return True
    
    def _on_rollout_end(self) -> bool:
        """Called at the end of each rollout (after policy update)"""
        
        # First, collect training metrics from DQN logger
        training_record = None
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            log_data = self.model.logger.name_to_value
            
            # Create training record
            training_record = {
                'timestep': self.num_timesteps,
                'loss': log_data.get('train/loss', np.nan),
                'q_value_mean': log_data.get('train/q_value', np.nan),
                'exploration_rate': getattr(self.model, 'exploration_rate', np.nan),
                'target_updates': getattr(self.model, '_n_updates', 0) // getattr(self.model, 'target_update_interval', 1)
            }
            
            # Calculate loss variance window
            recent_losses = [tr['loss'] for tr in self.training_data[-10:] if not np.isnan(tr.get('loss', np.nan))]
            if len(recent_losses) > 2:
                training_record['loss_variance_window'] = np.var(recent_losses)
            else:
                training_record['loss_variance_window'] = np.nan
            
            self.training_data.append(training_record)
        
        # Process any episodes that completed during this rollout
        for i, (episode_reward, episode_length, success_metrics) in enumerate(zip(
            self.temp_episode_rewards, self.temp_episode_lengths, self.temp_episode_success_metrics)):
            
            # Update counters
            self.episode_count += 1
            self.cumulative_reward_sum += episode_reward
            self.recent_rewards.append(episode_reward)
            
            # Calculate moving averages
            moving_avg_10 = np.mean(self.recent_rewards[-10:]) if len(self.recent_rewards) >= 10 else np.mean(self.recent_rewards)
            moving_avg_100 = np.mean(self.recent_rewards[-100:]) if len(self.recent_rewards) >= 100 else np.mean(self.recent_rewards)
            
            # Create episode record with training metrics included
            episode_record = {
                'episode_num': self.episode_count,
                'timestep': self.num_timesteps,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'cumulative_reward': self.cumulative_reward_sum,
                'moving_avg_10': moving_avg_10,
                'moving_avg_100': moving_avg_100,
                'conditions_read': success_metrics.get('conditions_read', False),
                'navigation_started': success_metrics.get('started_navigation', False),
                'zones_reached': success_metrics.get('reached_any_zone', False),
                'correct_zones': success_metrics.get('chose_correct_zone', False),
                'treatments_applied': success_metrics.get('applied_treatment', False),
                'episode_phases': success_metrics.get('phase_reached', 'UNKNOWN'),
                'final_distances': success_metrics.get('final_distance_to_target', 0),
                'best_distances': success_metrics.get('best_distance_achieved', 0),
                # ADD TRAINING METRICS TO EPISODE RECORDS
                'loss': training_record['loss'] if training_record else np.nan,
                'q_value_mean': training_record['q_value_mean'] if training_record else np.nan,
                'exploration_rate': training_record['exploration_rate'] if training_record else np.nan,
                'target_updates': training_record['target_updates'] if training_record else np.nan,
                'loss_variance_window': training_record['loss_variance_window'] if training_record else np.nan
            }
            
            self.episode_data.append(episode_record)
            
            if self.verbose >= 1:
                print(f"✓ Episode {self.episode_count}: Reward={episode_reward:.2f}, "
                      f"Cumulative={self.cumulative_reward_sum:.2f}, MA100={moving_avg_100:.2f}, "
                      f"Eps={training_record['exploration_rate']:.3f}" if training_record else "")
        
        # Clear temporary storage
        self.temp_episode_rewards = []
        self.temp_episode_lengths = []
        self.temp_episode_success_metrics = []
        
        # Save metrics periodically
        if self.num_timesteps % self.save_freq == 0:
            self.save_metrics_csv(f"checkpoint_{self.num_timesteps}")
        
        return True
    
    def save_metrics_csv(self, filename_suffix=""):
        """Save all collected metrics as CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save episode metrics
        if self.episode_data:
            episode_df = pd.DataFrame(self.episode_data)
            episode_file = f"dqn_episode_metrics_{filename_suffix}_{timestamp}.csv"
            episode_path = os.path.join(self.save_path, episode_file)
            episode_df.to_csv(episode_path, index=False)
            saved_files['episode_metrics'] = episode_path
            print(f"DQN Episode metrics saved to: {episode_path}")
            
            # saving the key metrics + training metrics
            simple_episode_df = episode_df[['episode_num', 'episode_reward', 'cumulative_reward', 
                                          'moving_avg_10', 'moving_avg_100', 'episode_length',
                                          'loss', 'q_value_mean', 'exploration_rate', 'target_updates', 
                                          'loss_variance_window']].copy()
            simple_file = f"dqn_episode_rewards_{filename_suffix}_{timestamp}.csv"
            simple_path = os.path.join(self.save_path, simple_file)
            simple_episode_df.to_csv(simple_path, index=False)
            saved_files['episode_rewards'] = simple_path
            print(f"DQN Simplified episode rewards saved to: {simple_path}")
        
        # Save training metrics
        if self.training_data:
            training_df = pd.DataFrame(self.training_data)
            training_file = f"dqn_training_metrics_{filename_suffix}_{timestamp}.csv"
            training_path = os.path.join(self.save_path, training_file)
            training_df.to_csv(training_path, index=False)
            saved_files['training_metrics'] = training_path
            print(f"DQN Training metrics saved to: {training_path}")
        
        # Save summary statistics
        if self.episode_data:
            episode_rewards = [ep['episode_reward'] for ep in self.episode_data]
            recent_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            
            # Calculate success metrics
            success_rates = {}
            if self.episode_data:
                success_rates = {
                    'conditions_read_rate': np.mean([ep['conditions_read'] for ep in self.episode_data[-100:]]) * 100,
                    'navigation_started_rate': np.mean([ep['navigation_started'] for ep in self.episode_data[-100:]]) * 100,
                    'zones_reached_rate': np.mean([ep['zones_reached'] for ep in self.episode_data[-100:]]) * 100,
                    'correct_zones_rate': np.mean([ep['correct_zones'] for ep in self.episode_data[-100:]]) * 100,
                    'treatments_applied_rate': np.mean([ep['treatments_applied'] for ep in self.episode_data[-100:]]) * 100
                }
            
            summary_data = {
                'metric': [
                    'total_episodes',
                    'total_timesteps',
                    'final_cumulative_reward',
                    'mean_episode_reward',
                    'std_episode_reward',
                    'min_episode_reward',
                    'max_episode_reward',
                    'mean_episode_length',
                    'final_moving_avg_10',
                    'final_moving_avg_100',
                    'reward_variance_last_100',
                    'mean_reward_last_100',
                    'conditions_read_success_rate',
                    'navigation_started_success_rate', 
                    'zones_reached_success_rate',
                    'correct_zones_success_rate',
                    'treatments_applied_success_rate',
                    'final_exploration_rate',
                    'final_loss',
                    'final_q_value_mean',
                    'timestamp'
                ],
                'value': [
                    len(episode_rewards),
                    self.num_timesteps,
                    self.cumulative_reward_sum,
                    np.mean(episode_rewards),
                    np.std(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    np.mean([ep['episode_length'] for ep in self.episode_data]),
                    self.episode_data[-1]['moving_avg_10'] if self.episode_data else 0,
                    self.episode_data[-1]['moving_avg_100'] if self.episode_data else 0,
                    np.var(recent_100_rewards) if len(recent_100_rewards) > 1 else 0,
                    np.mean(recent_100_rewards),
                    success_rates.get('conditions_read_rate', 0),
                    success_rates.get('navigation_started_rate', 0),
                    success_rates.get('zones_reached_rate', 0),
                    success_rates.get('correct_zones_rate', 0),
                    success_rates.get('treatments_applied_rate', 0),
                    self.episode_data[-1]['exploration_rate'] if self.episode_data else np.nan,
                    self.episode_data[-1]['loss'] if self.episode_data else np.nan,
                    self.episode_data[-1]['q_value_mean'] if self.episode_data else np.nan,
                    timestamp
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = f"dqn_training_summary_{filename_suffix}_{timestamp}.csv"
            summary_path = os.path.join(self.save_path, summary_file)
            summary_df.to_csv(summary_path, index=False)
            saved_files['summary'] = summary_path
            print(f"DQN Training summary saved to: {summary_path}")
        
        return saved_files
    
    def get_current_stats(self):
        """Return current training statistics"""
        if not self.episode_data:
            return {}
        
        episode_rewards = [ep['episode_reward'] for ep in self.episode_data]
        return {
            'total_episodes': len(episode_rewards),
            'cumulative_reward': self.cumulative_reward_sum,
            'mean_reward': np.mean(episode_rewards),
            'recent_100_mean': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
            'current_moving_avg_10': self.episode_data[-1]['moving_avg_10'],
            'current_moving_avg_100': self.episode_data[-1]['moving_avg_100'],
            'current_exploration_rate': self.episode_data[-1]['exploration_rate'] if self.episode_data else np.nan
        }

class CustomMonitor(Monitor):
    """Custom Monitor wrapper that ensures episode info is properly passed"""
    
    def __init__(self, env, filename=None, allow_early_resets=True, reset_keywords=(), info_keywords=()):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)
        # Properly initializing episode tracking variables
        self.episode_returns = 0.0
        self.episode_lengths = 0
        self.t_start = time.time()
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.episode_returns += float(reward)
        self.episode_lengths += 1
        
        if terminated or truncated:
            # Get success metrics from environment if available
            success_metrics = {}
            if hasattr(self.env, 'get_success_metrics'):
                success_metrics = self.env.get_success_metrics()
            
            # Create episode info
            episode_info = {
                'episode': {
                    'r': float(self.episode_returns),
                    'l': int(self.episode_lengths),
                    't': time.time() - self.t_start
                },
                'success_metrics': success_metrics
            }
            
            if info is None:
                info = {}
            info.update(episode_info)
            
            # Reset episode counters
            self.episode_returns = 0.0
            self.episode_lengths = 0
            self.t_start = time.time()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment and episode counters"""
        obs, info = self.env.reset(**kwargs)
        self.episode_returns = 0.0
        self.episode_lengths = 0
        self.t_start = time.time()
        return obs, info

def create_storage_env(config=None):
    """Create and wrap the storage environment"""
    if config is None:
        config = {
            'max_steps': 25,
            'initial_pest': 0.1,
            'location': "Kigali",
            'grid_size': (5, 5),
            'layout': 'custom',
            'obs_type': 'mlp',
            'curriculum_stage': 1,
            'use_action_masking': False,
            'intermediate_rewards': True
        }
    
    env = StorageEnv(config=config, render_mode=None, obs_type='mlp')
    env = CustomMonitor(env)
    return env

def train_dqn(total_timesteps=100000, config=None, save_path="./models/dqn/"):
    """Training DQN agent on storage environment"""
    
    print("=" * 60)
    print("TRAINING DQN AGENT")
    print("=" * 60)
    
    # Create environment
    env = create_storage_env(config)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create logs directory
    log_path = "./logs/dqn/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Setup callback for metrics collection
    callback = DQNMetricsCallback(save_path=log_path, save_freq=5000, verbose=1)
    
    # DQN Hyperparameters
    dqn_params = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'exploration_fraction': 0.5,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.1,
        'buffer_size': 100000,
        'batch_size': 32,
        'learning_starts': 5000,
        'target_update_interval': 300,
        'train_freq': 4,
        'policy_kwargs': {
            'net_arch': [256, 256, 128],
            'activation_fn': nn.ReLU
        },
        'verbose': 1,
        'seed': 42
    }
    
    print("DQN Hyperparameters:")
    for key, value in dqn_params.items():
        if key != 'policy_kwargs':
            print(f"  {key}: {value}")
    print(f"  network_architecture: {dqn_params['policy_kwargs']['net_arch']}")
    print(f"  activation_function: {dqn_params['policy_kwargs']['activation_fn'].__name__}")
    
    # Create DQN model
    model = DQN(
        'MlpPolicy',
        vec_env,
        **dqn_params,
        tensorboard_log=log_path
    )
    
    print(f"\nStarting training for {total_timesteps} timesteps...")
    # Access environment attributes through the Monitor wrapper
    underlying_env = env.env if hasattr(env, 'env') else env
    print(f"Environment: {underlying_env.grid_size} grid, {len(underlying_env.ACTIONS)} actions")
    print(f"Observation space: {env.observation_space}")
    
    # Train the model
    start_time = datetime.now()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    training_time = datetime.now() - start_time
    
    print(f"\nTraining completed in {training_time}")
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"dqn_storage_model_{timestamp}")
    model.save(model_path)
    
    # Save hyperparameters
    hyperparams_path = os.path.join(save_path, f"dqn_hyperparams_{timestamp}.json")
    with open(hyperparams_path, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_params = {}
        for key, value in dqn_params.items():
            if key == 'policy_kwargs':
                serializable_params[key] = {
                    'net_arch': value['net_arch'],
                    'activation_fn': value['activation_fn'].__name__
                }
            else:
                serializable_params[key] = value
        
        json.dump({
            'hyperparameters': serializable_params,
            'total_timesteps': total_timesteps,
            'training_time_seconds': training_time.total_seconds(),
            'timestamp': timestamp
        }, f, indent=2)
    
    # Save final metrics
    final_csv_paths = callback.save_metrics_csv(f"final_{timestamp}")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Hyperparameters saved to: {hyperparams_path}")
    print("Final CSV files saved:")
    for csv_type, path in final_csv_paths.items():
        if path:
            print(f"  {csv_type}: {path}")
    
    # Print training summary
    if callback.episode_data:
        episode_rewards = [ep['episode_reward'] for ep in callback.episode_data]
        print(f"\nTraining Summary:")
        print(f"  Total episodes: {len(episode_rewards)}")
        print(f"  Final cumulative reward: {callback.cumulative_reward_sum:.2f}")
        print(f"  Average reward: {np.mean(episode_rewards[-100:]):.2f} (last 100)")
        print(f"  Max reward: {np.max(episode_rewards):.2f}")
        
        if callback.episode_data:
            success_rate = np.mean([ep['correct_zones'] for ep in callback.episode_data[-100:]]) * 100
            print(f"  Success rate: {success_rate:.1f}% (last 100 episodes)")
        
        if callback.training_data and not np.isnan(callback.training_data[-1]['loss']):
            print(f"  Final loss: {callback.training_data[-1]['loss']:.4f}")
        if callback.training_data and not np.isnan(callback.training_data[-1]['exploration_rate']):
            print(f"  Final exploration rate: {callback.training_data[-1]['exploration_rate']:.4f}")
        if callback.training_data and not np.isnan(callback.training_data[-1]['q_value_mean']):
            print(f"  Final Q-value mean: {callback.training_data[-1]['q_value_mean']:.4f}")
            
        # Convergence analysis
        recent_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        reward_variance = np.var(recent_100_rewards) if len(recent_100_rewards) > 1 else float('inf')
        print(f"  Reward stability (variance): {reward_variance:.4f}")
        
        if reward_variance < 0.1:
            print(" Training appears to have converged (low reward variance)")
        else:
            print(" Training may need more episodes to converge")
    
    env.close()
    return model, callback, model_path

if __name__ == "__main__":
    # Environment configuration
    env_config = {
        'max_steps': 25,
        'initial_pest': 0.1,
        'location': "Kigali", 
        'grid_size': (5, 5),
        'layout': 'custom',
        'obs_type': 'mlp',
        'curriculum_stage': 2,  # Start with medium difficulty
        'use_action_masking': False,
        'intermediate_rewards': True
    }
    
    # Train DQN
    model, callback, model_path = train_dqn(
        total_timesteps=250000,  # Increased for better learning
        config=env_config,
        save_path="./models/dqn/"
    )
    
    print("DQN training completed successfully!")