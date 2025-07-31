import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import gymnasium as gym
import numpy as np
import torch
import pandas as pd
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from environment.custom_env import StorageEnv

# === Settings ===
SAVE_DIR = "logs/ppo_run"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Custom Callback for Logging ===
class PPOLoggingCallback(BaseCallback):
    """
    Custom callback for PPO training with detailed logging
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.entropies = []
        self.timesteps = []
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.clip_fractions = []
        self.kl_divergences = []

    def _on_step(self) -> bool:
        # Check if episode ended and log rewards
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and info['episode'] is not None:
                    self.episode_rewards.append(info['episode']['r'])
                    if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                        recent_mean = np.mean(self.episode_rewards[-10:])
                        print(f"Episode {len(self.episode_rewards)}: Recent 10-episode mean reward: {recent_mean:.2f}")

        # Log PPO-specific metrics
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            entropy = self.model.logger.name_to_value.get("train/entropy_loss", 0.0)
            policy_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss", 0.0)
            value_loss = self.model.logger.name_to_value.get("train/value_loss", 0.0)
            clip_fraction = self.model.logger.name_to_value.get("train/clip_fraction", 0.0)
            kl_div = self.model.logger.name_to_value.get("train/approx_kl", 0.0)
            
            self.entropies.append(entropy)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.clip_fractions.append(clip_fraction)
            self.kl_divergences.append(kl_div)
            self.timesteps.append(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Save metrics
        max_len = min(len(self.timesteps), len(self.entropies))
        if max_len > 0:
            df = pd.DataFrame({
                'Timestep': self.timesteps[:max_len],
                'Entropy': self.entropies[:max_len],
                'Policy_Loss': self.policy_losses[:max_len],
                'Value_Loss': self.value_losses[:max_len],
                'Clip_Fraction': self.clip_fractions[:max_len],
                'KL_Divergence': self.kl_divergences[:max_len]
            })
            df.to_csv(os.path.join(SAVE_DIR, "training_metrics.csv"), index=False)
        
        # Save episode rewards separately
        if self.episode_rewards:
            reward_df = pd.DataFrame({
                'Episode': range(1, len(self.episode_rewards) + 1),
                'Reward': self.episode_rewards
            })
            reward_df.to_csv(os.path.join(SAVE_DIR, "episode_rewards.csv"), index=False)

# === Helper function for robust action conversion ===
def convert_action_to_int(action):
    """
    Robustly convert action from model.predict() to integer
    Handles both 0-dimensional and multi-dimensional numpy arrays
    """
    if isinstance(action, np.ndarray):
        if action.ndim == 0:  # 0-dimensional array (scalar)
            return int(action.item())
        else:  # 1 or more dimensional array
            return int(action.flatten()[0])
    else:
        return int(action)  # Already a scalar

# === Create Environment ===
def make_env(rank: int = 0, seed: int = 0):
    """Create environment with proper seeding for parallel training"""
    def _init():
        config = {
            'max_days': 30,
            'initial_pest': 0.1,
            'location': "Kigali",
            'grid_size': (5, 5),
            'layout': 'custom',
            'obs_type': 'mlp',
            'curriculum_level': 1  # Start with medium difficulty
        }
        env = StorageEnv(config=config, obs_type="mlp")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # Test environment creation
    print("Testing environment creation...")
    test_env = make_env(0, 0)()
    obs, info = test_env.reset()
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    print(f"Sample observation shape: {obs.shape}")
    print(f"Sample observation: {obs}")

    # FIX: Access ACTIONS from the underlying environment, not the Monitor wrapper
    # The Monitor wrapper doesn't expose ACTIONS, but the underlying env does
    underlying_env = test_env.env if hasattr(test_env, 'env') else test_env
    while hasattr(underlying_env, 'env'):  # Navigate through all wrappers
        underlying_env = underlying_env.env
    
    # Now underlying_env should be the actual StorageEnv instance
    print(f"Environment type: {type(underlying_env)}")
    print(f"Available actions: {len(underlying_env.ACTIONS)}")

    # Create vectorized environment
    # FIXED: Use DummyVecEnv for Windows compatibility instead of SubprocVecEnv
    # If you want to use parallel environments on Windows, you need proper multiprocessing protection
    n_envs = 4
    print(f"Creating {n_envs} parallel environments...")
    
    # Use DummyVecEnv which runs sequentially but is more stable on Windows
    env = DummyVecEnv([make_env(i, 42) for i in range(n_envs)])
    
    # Alternative: If you want true parallel processing on Windows, uncomment this:
    # But you need to ensure this script is run with if __name__ == '__main__': protection
    # env = SubprocVecEnv([make_env(i, 42) for i in range(n_envs)])

    # === Hyperparameters and Model Init ===
    print("Initializing PPO model...")
    # PPO hyperparameters optimized for your discrete action environment
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,         # Standard PPO learning rate
        gamma=0.99,                 # Discount factor
        gae_lambda=0.95,           # GAE parameter for advantage estimation
        clip_range=0.2,            # PPO clipping parameter
        clip_range_vf=None,        # No value function clipping
        ent_coef=0.01,             # Entropy coefficient for exploration
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        n_steps=2048,              # Steps per environment per update
        batch_size=64,             # Minibatch size
        n_epochs=10,               # Number of epochs per update
        target_kl=0.01,            # Target KL divergence
        policy_kwargs=dict(
            net_arch=[256, 256],   # Neural network architecture
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log="./tensorboard_logs/PPO"
    )

    print("Model initialized successfully!")
    print(f"Policy: {model.policy}")
    print(f"Observation space: {model.observation_space}")
    print(f"Using {n_envs} parallel environments")

    # === Train ===
    print("Starting PPO training...")
    callback = PPOLoggingCallback(check_freq=1000)
    start_time = time.time()

    try:
        model.learn(total_timesteps=200000, callback=callback)  # More timesteps for PPO
        model.save(os.path.join(SAVE_DIR, "ppo_storage_model"))
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # === Evaluate and Save Results ===
    print("Starting evaluation...")
    eval_env = make_env(0, 123)()  # Different seed for evaluation
    episode_rewards = []
    episode_lengths = []
    final_pest_levels = []
    successful_episodes = []
    action_distributions = []

    for episode in range(20):  # More episodes for better statistics
        obs, info = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_actions = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            # Use robust action conversion
            action = convert_action_to_int(action)
            
            episode_actions.append(action)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            # Print some episode info
            if steps % 10 == 0:
                print(f"Episode {episode+1}, Step {steps}: Action={action}, Reward={reward:.2f}, "
                      f"Pest Level={info.get('pest_level', 'N/A'):.3f}, "
                      f"Zone={info.get('zone_type', 'N/A')}")
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_pest_level = info.get('pest_level', 0)
        final_pest_levels.append(final_pest_level)
        successful_episodes.append(1 if final_pest_level < 0.8 else 0)
        action_distributions.append(episode_actions)
        
        print(f"Episode {episode+1} finished: Reward={total_reward:.2f}, "
              f"Steps={steps}, Final Pest Level={final_pest_level:.3f}")

    # FIX: Calculate action usage statistics using the underlying environment
    all_actions = [action for episode_actions in action_distributions for action in episode_actions]
    action_counts = np.bincount(all_actions, minlength=len(underlying_env.ACTIONS))
    action_percentages = action_counts / len(all_actions) * 100

    # Calculate and save evaluation metrics
    mean_eval_reward = np.mean(episode_rewards)
    std_eval_reward = np.std(episode_rewards)
    mean_episode_length = np.mean(episode_lengths)
    mean_final_pest = np.mean(final_pest_levels)
    success_rate = np.mean(successful_episodes) * 100

    eval_results = {
        'mean_reward': mean_eval_reward,
        'std_reward': std_eval_reward,
        'mean_length': mean_episode_length,
        'mean_final_pest': mean_final_pest,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_pest_levels': final_pest_levels,
        'action_counts': action_counts.tolist(),
        'action_percentages': action_percentages.tolist()
    }

    # Save evaluation results
    with open(os.path.join(SAVE_DIR, "eval_results.txt"), "w") as f:
        f.write(f"=== PPO Evaluation Results ===\n")
        f.write(f"Mean Reward: {mean_eval_reward:.3f} ± {std_eval_reward:.3f}\n")
        f.write(f"Mean Episode Length: {mean_episode_length:.1f} steps\n")
        f.write(f"Mean Final Pest Level: {mean_final_pest:.3f}\n")
        f.write(f"Success Rate (Pest < 0.8): {success_rate:.1f}%\n")
        f.write(f"Training Time: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Total Timesteps: 200000\n")
        f.write(f"Parallel Environments: {n_envs}\n")
        f.write(f"\nAction Usage Distribution:\n")
        for i, (count, percentage) in enumerate(zip(action_counts, action_percentages)):
            if i < len(underlying_env.ACTIONS):
                f.write(f"Action {i} ({underlying_env.ACTIONS[i]}): {count} times ({percentage:.1f}%)\n")
        f.write(f"\nIndividual Episode Results:\n")
        for i, (reward, length, pest) in enumerate(zip(episode_rewards, episode_lengths, final_pest_levels)):
            f.write(f"Episode {i+1}: Reward={reward:.2f}, Length={length}, Final Pest={pest:.3f}\n")

    # Save as CSV for analysis
    eval_df = pd.DataFrame({
        'Episode': range(1, len(episode_rewards) + 1),
        'Reward': episode_rewards,
        'Length': episode_lengths,
        'Final_Pest_Level': final_pest_levels,
        'Success': successful_episodes
    })
    eval_df.to_csv(os.path.join(SAVE_DIR, "evaluation_results.csv"), index=False)

    # Save action usage statistics
    action_df = pd.DataFrame({
        'Action_Index': range(len(action_counts)),
        'Action_Name': [underlying_env.ACTIONS[i] if i < len(underlying_env.ACTIONS) else f"Action_{i}" for i in range(len(action_counts))],
        'Count': action_counts,
        'Percentage': action_percentages
    })
    action_df.to_csv(os.path.join(SAVE_DIR, "action_usage.csv"), index=False)

    print(f"\n✅ PPO Training Complete!")
    print(f"📊 Evaluation Results:")
    print(f"   Mean Reward: {mean_eval_reward:.3f} ± {std_eval_reward:.3f}")
    print(f"   Mean Episode Length: {mean_episode_length:.1f} steps")
    print(f"   Mean Final Pest Level: {mean_final_pest:.3f}")
    print(f"   Success Rate (Pest < 0.8): {success_rate:.1f}%")
    print(f"📊 Top 5 Most Used Actions:")
    top_actions = sorted(enumerate(action_percentages), key=lambda x: x[1], reverse=True)[:5]
    for idx, percentage in top_actions:
        if idx < len(underlying_env.ACTIONS):
            print(f"   {underlying_env.ACTIONS[idx]}: {percentage:.1f}%")
    print(f"📁 Results saved to: {SAVE_DIR}")

    # Clean up
    eval_env.close()
    env.close()
    test_env.close()

if __name__ == "__main__":
    freeze_support()  # Required for Windows multiprocessing
    main()