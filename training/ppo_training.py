import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from stable_baselines3 import PPO
from training.base_train import (
    setup_environment,
    get_device,
    create_callbacks,
    set_global_seeds
)

def train_ppo(config=None, total_timesteps=500_000):
    # Initialize reproducibility
    set_global_seeds(config.get("seed", 42))
    device = get_device()
    print(f"Training PPO on: {device}")

    # Environment setup with seed control
    env = setup_environment(config, seed=config.get("seed", 42))
    eval_env = setup_environment(config, seed=config.get("seed", 42))

    # Enhanced network architecture
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128],  # Policy network
            "vf": [128, 128]   # Value network
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
    }

    # Optimized PPO configuration
    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy for Dict observations
        env,
        device=device,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,        # Increased from 1024 for better advantage estimates
        batch_size=64,        # Increased from 32 for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/ppo/",
        seed=config.get("seed", 42)
    )

    # Unified callback system
    callbacks = create_callbacks(
        log_dir="./models/ppo",
        eval_env=eval_env,
        save_freq=10_000,
        eval_freq=5_000
    )

    # Training execution
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_run",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # Final model save
    model.save("./models/ppo/ppo_final")
    if hasattr(env, "save_running_average"):
        env.save_running_average("./models/ppo/vecnormalize.pkl")
    env.close()

if __name__ == "__main__":
    config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali",
        'grid_size': (5, 5),
        'layout': 'custom',
        'render_mode': None,
        'seed': 42  # Added explicit seed
    }
    train_ppo(config=config, total_timesteps=1_000_000)