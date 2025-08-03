import os
import sys
# Adding project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch.nn as nn
from environment.custom_env import StorageEnv

class PPOMetricsCallback(BaseCallback):
    """callback class to collect PPO training metrics and save as CSV files"""
    
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
        """Called at each environment step"""
        
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
        """Called at the end of each rollout (after policy update) to process completed episodes rewards"""
        
        # First, collect training metrics from PPO logger
        training_record = None
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            log_data = self.model.logger.name_to_value
            
            # Get current learning rate
            current_lr = None
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    current_lr = self.model.learning_rate(1.0)
                else:
                    current_lr = self.model.learning_rate
            
            # Create training record
            training_record = {
                'timestep': self.num_timesteps,
                'policy_loss': log_data.get('train/policy_gradient_loss', np.nan),
                'value_loss': log_data.get('train/value_loss', np.nan),
                'entropy': -log_data.get('train/entropy_loss', np.nan) if 'train/entropy_loss' in log_data else np.nan,
                'clip_fraction': log_data.get('train/clip_fraction', np.nan),
                'kl_divergence': log_data.get('train/approx_kl', np.nan),
                'learning_rate': current_lr if current_lr is not None else np.nan,
                'explained_variance': log_data.get('train/explained_variance', np.nan)
            }
            
            # Calculate total loss and add derived metrics
            if not np.isnan(training_record['policy_loss']) and not np.isnan(training_record['value_loss']):
                training_record['total_loss'] = training_record['policy_loss'] + training_record['value_loss']
            else:
                training_record['total_loss'] = np.nan
            
            # Calculate loss variance window
            recent_losses = [tr['total_loss'] for tr in self.training_data[-10:] if not np.isnan(tr.get('total_loss', np.nan))]
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
                'policy_loss': training_record['policy_loss'] if training_record else np.nan,
                'value_loss': training_record['value_loss'] if training_record else np.nan,
                'entropy': training_record['entropy'] if training_record else np.nan,
                'clip_fraction': training_record['clip_fraction'] if training_record else np.nan,
                'kl_divergence': training_record['kl_divergence'] if training_record else np.nan,
                'learning_rate': training_record['learning_rate'] if training_record else np.nan,
                'explained_variance': training_record['explained_variance'] if training_record else np.nan,
                'total_loss': training_record['total_loss'] if training_record else np.nan,
                'loss_variance_window': training_record['loss_variance_window'] if training_record else np.nan
            }
            
            self.episode_data.append(episode_record)
            
            if self.verbose >= 1:
                print(f"✓ Episode {self.episode_count}: Reward={episode_reward:.2f}, "
                      f"Cumulative={self.cumulative_reward_sum:.2f}, MA100={moving_avg_100:.2f}, "
                      f"ClipFrac={training_record['clip_fraction']:.3f}" if training_record else "")
        
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
            episode_file = f"ppo_episode_metrics_{filename_suffix}_{timestamp}.csv"
            episode_path = os.path.join(self.save_path, episode_file)
            episode_df.to_csv(episode_path, index=False)
            saved_files['episode_metrics'] = episode_path
            print(f"PPO Episode metrics saved to: {episode_path}")
            
            # save the key metrics + training metrics
            simple_episode_df = episode_df[['episode_num', 'episode_reward', 'cumulative_reward', 
                                          'moving_avg_10', 'moving_avg_100', 'episode_length',
                                          'policy_loss', 'value_loss', 'entropy', 'clip_fraction',
                                          'kl_divergence', 'learning_rate', 'explained_variance', 
                                          'total_loss', 'loss_variance_window']].copy()
            simple_file = f"ppo_episode_rewards_{filename_suffix}_{timestamp}.csv"
            simple_path = os.path.join(self.save_path, simple_file)
            simple_episode_df.to_csv(simple_path, index=False)
            saved_files['episode_rewards'] = simple_path
            print(f"PPO Simplified episode rewards saved to: {simple_path}")
        
        # Save training metrics
        if self.training_data:
            training_df = pd.DataFrame(self.training_data)
            training_file = f"ppo_training_metrics_{filename_suffix}_{timestamp}.csv"
            training_path = os.path.join(self.save_path, training_file)
            training_df.to_csv(training_path, index=False)
            saved_files['training_metrics'] = training_path
            print(f"PPO Training metrics saved to: {training_path}")
        
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
                    'final_policy_loss',
                    'final_entropy',
                    'final_clip_fraction',
                    'final_kl_divergence',
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
                    self.episode_data[-1]['policy_loss'] if self.episode_data else np.nan,
                    self.episode_data[-1]['entropy'] if self.episode_data else np.nan,
                    self.episode_data[-1]['clip_fraction'] if self.episode_data else np.nan,
                    self.episode_data[-1]['kl_divergence'] if self.episode_data else np.nan,
                    timestamp
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = f"ppo_training_summary_{filename_suffix}_{timestamp}.csv"
            summary_path = os.path.join(self.save_path, summary_file)
            summary_df.to_csv(summary_path, index=False)
            saved_files['summary'] = summary_path
            print(f"PPO Training summary saved to: {summary_path}")
        
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
            'current_clip_fraction': self.episode_data[-1]['clip_fraction'] if self.episode_data else np.nan
        }

class CustomMonitor(Monitor):
    """Custom Monitor wrapper that ensures episode info is properly passed"""
    
    def __init__(self, env, filename=None, allow_early_resets=True, reset_keywords=(), info_keywords=()):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)
        # Properly initialize episode tracking variables
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
            
            # Add to info dict
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

def train_ppo(total_timesteps=100000, config=None, save_path="./models/ppo/"):
    """Trainining PPO agent on storage environment"""
    
    print("=" * 60)
    print("TRAINING PPO AGENT")
    print("=" * 60)
    
    # Create environment
    env = create_storage_env(config)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create logs directory
    log_path = "./logs/ppo/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Setup callback for metrics collection
    callback = PPOMetricsCallback(save_path=log_path, save_freq=5000, verbose=1)
    
    # PPO Hyperparameters
    ppo_params = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.15,
        'clip_range_vf': 0.2,
        'ent_coef': 0.03,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'target_kl': 0.01,
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),
            'activation_fn': nn.Tanh
        },
        'verbose': 1,
        'seed': 42
    }
    
    print("PPO Hyperparameters:")
    for key, value in ppo_params.items():
        if key != 'policy_kwargs':
            print(f"  {key}: {value}")
    print(f"  policy_network: {ppo_params['policy_kwargs']['net_arch']['pi']}")
    print(f"  value_network: {ppo_params['policy_kwargs']['net_arch']['vf']}")
    print(f"  activation_function: {ppo_params['policy_kwargs']['activation_fn'].__name__}")
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        vec_env,
        **ppo_params,
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
    model_path = os.path.join(save_path, f"ppo_storage_model_{timestamp}")
    model.save(model_path)
    
    # Save hyperparameters
    hyperparams_path = os.path.join(save_path, f"ppo_hyperparams_{timestamp}.json")
    with open(hyperparams_path, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_params = {}
        for key, value in ppo_params.items():
            if key == 'policy_kwargs':
                serializable_params[key] = {
                    'net_arch_policy': value['net_arch']['pi'],
                    'net_arch_value': value['net_arch']['vf'],
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
        
        if callback.training_data and not np.isnan(callback.training_data[-1]['policy_loss']):
            print(f"  Final policy loss: {callback.training_data[-1]['policy_loss']:.4f}")
        if callback.training_data and not np.isnan(callback.training_data[-1]['entropy']):
            print(f"  Final entropy: {callback.training_data[-1]['entropy']:.4f}")
        if callback.training_data and not np.isnan(callback.training_data[-1]['clip_fraction']):
            print(f"  Final clip fraction: {callback.training_data[-1]['clip_fraction']:.4f}")
        if callback.training_data and not np.isnan(callback.training_data[-1]['kl_divergence']):
            print(f"  Final KL divergence: {callback.training_data[-1]['kl_divergence']:.4f}")
            
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
        'curriculum_stage': 2,
        'use_action_masking': True,
        'intermediate_rewards': True
    }
    
    # Train PPO
    model, callback, model_path = train_ppo(
        total_timesteps=350000,
        config=env_config,
        save_path="./models/ppo/"
    )
    
    print("PPO training completed successfully!")