import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import torch
from stable_baselines3 import A2C
from training.base_train import (
    setup_environment,
    get_device,
    create_callbacks,
    set_global_seeds
)

def train_a2c(config=None, total_timesteps=500_000):
    # Initialize reproducibility
    set_global_seeds(config.get("seed", 42))
    device = get_device()
    print(f"Training on: {device}")
    
    # Environment setup
    env = setup_environment(config, seed=config.get("seed", 42))
    eval_env = setup_environment(config, seed=config.get("seed", 42))
    
    # Model configuration
    model = A2C(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
        learning_rate=6e-4,
        n_steps=32,
        gamma=0.99,
        gae_lambda=0.9,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs={
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True,
            "share_features_extractor": True
        },
        tensorboard_log="./logs/a2c/",
        seed=config.get("seed", 42)  # Consistent seeding
    )
    
    # Training execution
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=create_callbacks(
                log_dir="./models/a2c",
                eval_env=eval_env,
                save_freq=10000,
                eval_freq=5000
            ),
            tb_log_name="a2c_run",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
    
    # Save final artifacts
    model.save("./models/a2c/a2c_final")
    if hasattr(env, 'save_running_average'):
        env.save_running_average("./models/a2c/vecnormalize.pkl")
    env.close()

if __name__ == "__main__":
    config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali",
        'grid_size': (5, 5),
        'layout': 'custom',
        'render_mode': None,
        'seed': 42  # Explicit seed in config
    }
    train_a2c(config=config, total_timesteps=1_000_000)