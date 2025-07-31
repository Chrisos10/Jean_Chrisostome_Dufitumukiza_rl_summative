import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import gymnasium as gym
import numpy as np
import torch
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import StorageEnv  # Adjust to your actual path

# === Settings ===
SAVE_DIR = "logs/reinforce_run"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Custom Callback for Logging ===
class REINFORCELoggingCallback(BaseCallback):
    """
    A2C configured to behave like REINFORCE (policy gradient)
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

    def _on_step(self) -> bool:
        # Check if episode ended and log rewards
        if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
            episode_info = self.locals['infos'][0]['episode']
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    recent_mean = np.mean(self.episode_rewards[-10:])
                    print(f"Episode {len(self.episode_rewards)}: Recent 10-episode mean reward: {recent_mean:.2f}")

        # Log REINFORCE-like metrics (using A2C's policy gradient components)
        entropy = self.model.logger.name_to_value.get("train/policy_entropy", 0.0)
        policy_loss = self.model.logger.name_to_value.get("train/policy_loss", 0.0)
        value_loss = self.model.logger.name_to_value.get("train/value_loss", 0.0)
        
        self.entropies.append(entropy)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.timesteps.append(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Save metrics
        max_len = min(len(self.timesteps), len(self.entropies))
        df = pd.DataFrame({
            'Timestep': self.timesteps[:max_len],
            'Entropy': self.entropies[:max_len],
            'Policy_Loss': self.policy_losses[:max_len],
            'Value_Loss': self.value_losses[:max_len]
        })
        df.to_csv(os.path.join(SAVE_DIR, "training_metrics.csv"), index=False)
        
        # Save episode rewards separately
        if self.episode_rewards:
            reward_df = pd.DataFrame({
                'Episode': range(1, len(self.episode_rewards) + 1),
                'Reward': self.episode_rewards
            })
            reward_df.to_csv(os.path.join(SAVE_DIR, "episode_rewards.csv"), index=False)

# === Create Environment ===
def make_env():
    # Set obs_type to "mlp" to use MlpPolicy
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
    return env

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

# Test environment creation
print("Testing environment creation...")
test_env = make_env()
obs, info = test_env.reset()
print(f"Observation space: {test_env.observation_space}")
print(f"Action space: {test_env.action_space}")
print(f"Sample observation shape: {obs.shape}")
print(f"Sample observation: {obs}")

# Create vectorized environment
env = DummyVecEnv([make_env])

# === Hyperparameters and Model Init ===
print("Initializing REINFORCE-like model using A2C...")
# Configure A2C to behave like REINFORCE:
# - n_steps = 1 for immediate updates after each step (like REINFORCE)
# - vf_coef = 0 to disable value function (pure policy gradient)
# - Higher entropy coefficient for exploration
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.003,      # Higher learning rate typical for REINFORCE
    gamma=0.99,
    n_steps=1,               # Update after each step like REINFORCE
    ent_coef=0.1,            # Higher entropy for exploration
    vf_coef=0.0,             # Disable value function (pure policy gradient)
    max_grad_norm=0.5,
    tensorboard_log="./tensorboard_logs/REINFORCE"
)

print("Model initialized successfully!")
print(f"Policy: {model.policy}")
print(f"Observation space: {model.observation_space}")
print(f"Configured as REINFORCE-like policy gradient method")

# === Train ===
print("Starting REINFORCE-like training...")
callback = REINFORCELoggingCallback(check_freq=1000)
start_time = time.time()

try:
    model.learn(total_timesteps=50000, callback=callback)
    model.save(os.path.join(SAVE_DIR, "reinforce_storage_model"))
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# === Evaluate and Save Results ===
print("Starting evaluation...")
eval_env = make_env()
episode_rewards = []
episode_lengths = []
final_pest_levels = []

for episode in range(10):
    obs, info = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        # Use robust action conversion
        action = convert_action_to_int(action)
        
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        steps += 1
        
        # Print some episode info
        if steps % 10 == 0:
            print(f"Episode {episode+1}, Step {steps}: Reward={reward:.2f}, "
                  f"Pest Level={info.get('pest_level', 'N/A'):.3f}, "
                  f"Zone={info.get('zone_type', 'N/A')}")
    
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    final_pest_levels.append(info.get('pest_level', 0))
    
    print(f"Episode {episode+1} finished: Reward={total_reward:.2f}, "
          f"Steps={steps}, Final Pest Level={final_pest_levels[-1]:.3f}")

# Calculate and save evaluation metrics
mean_eval_reward = np.mean(episode_rewards)
std_eval_reward = np.std(episode_rewards)
mean_episode_length = np.mean(episode_lengths)
mean_final_pest = np.mean(final_pest_levels)

eval_results = {
    'mean_reward': mean_eval_reward,
    'std_reward': std_eval_reward,
    'mean_length': mean_episode_length,
    'mean_final_pest': mean_final_pest,
    'episode_rewards': episode_rewards,
    'episode_lengths': episode_lengths,
    'final_pest_levels': final_pest_levels
}

# Save evaluation results
with open(os.path.join(SAVE_DIR, "eval_results.txt"), "w") as f:
    f.write(f"=== REINFORCE-like (A2C) Evaluation Results ===\n")
    f.write(f"Mean Reward: {mean_eval_reward:.3f} ± {std_eval_reward:.3f}\n")
    f.write(f"Mean Episode Length: {mean_episode_length:.1f} steps\n")
    f.write(f"Mean Final Pest Level: {mean_final_pest:.3f}\n")
    f.write(f"Success Rate (Pest < 0.8): {sum(1 for p in final_pest_levels if p < 0.8)/len(final_pest_levels)*100:.1f}%\n")
    f.write(f"\nIndividual Episode Results:\n")
    for i, (reward, length, pest) in enumerate(zip(episode_rewards, episode_lengths, final_pest_levels)):
        f.write(f"Episode {i+1}: Reward={reward:.2f}, Length={length}, Final Pest={pest:.3f}\n")

# Save as CSV for analysis
eval_df = pd.DataFrame({
    'Episode': range(1, len(episode_rewards) + 1),
    'Reward': episode_rewards,
    'Length': episode_lengths,
    'Final_Pest_Level': final_pest_levels
})
eval_df.to_csv(os.path.join(SAVE_DIR, "evaluation_results.csv"), index=False)

print(f"\n✅ REINFORCE-like Training Complete!")
print(f"📊 Evaluation Results:")
print(f"   Mean Reward: {mean_eval_reward:.3f} ± {std_eval_reward:.3f}")
print(f"   Mean Episode Length: {mean_episode_length:.1f} steps")
print(f"   Mean Final Pest Level: {mean_final_pest:.3f}")
print(f"   Success Rate (Pest < 0.8): {sum(1 for p in final_pest_levels if p < 0.8)/len(final_pest_levels)*100:.1f}%")
print(f"📁 Results saved to: {SAVE_DIR}")

# Clean up
eval_env.close()
env.close()