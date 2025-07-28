import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import os
import torch
from stable_baselines3 import A2C  # REINFORCE is A2C with n_steps=1
from training.base_train import (
    setup_environment,
    get_device,
    set_global_seeds,
    create_callbacks
)

def train_reinforce(config=None, total_timesteps=500_000):
    """Train REINFORCE using Stable Baselines3's A2C implementation"""
    # Initialize reproducibility
    set_global_seeds(config.get("seed", 42))
    device = get_device()
    print(f"Training REINFORCE (A2C-n_step=1) on: {device}")

    # Environment setup
    env = setup_environment(config, seed=config.get("seed", 42))
    eval_env = setup_environment(config, seed=config.get("seed", 42))

    # Policy network configuration
    policy_kwargs = {
        "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True
    }

    # REINFORCE = A2C with n_steps=1 (no advantage estimation)
    model = A2C(
        "MultiInputPolicy",
        env,
        n_steps=1,  # Key difference from standard A2C
        device=device,
        verbose=1,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=1.0,  # No advantage estimation
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        stats_window_size=1,  # Disables variance calculations
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/reinforce/",
        seed=config.get("seed", 42)
    )

    # Callbacks (same as other algorithms)
    callbacks = create_callbacks(
        log_dir="./models/reinforce",
        eval_env=eval_env,
        save_freq=10_000,
        eval_freq=5_000
    )

    # Training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="reinforce_run",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # Save final model
    model.save("./models/reinforce/reinforce_final")
    env.close()

if __name__ == "__main__":
    config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali",
        'grid_size': (5, 5),
        'layout': 'custom',
        'render_mode': None,
        'seed': 42
    }
    train_reinforce(config=config, total_timesteps=1_000_000)