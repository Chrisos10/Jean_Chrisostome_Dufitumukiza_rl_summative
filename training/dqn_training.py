# Add these lines at the very top
import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from training.base_train import (
    setup_environment,
    get_device,
    set_global_seeds,
    MetricsCallback  # Reuse this instead of CustomEvalCallback
)

def train_dqn(config=None, total_timesteps=500_000):
    # Set seeds for reproducibility
    set_global_seeds(config.get("seed", 42))
    device = get_device()
    print(f"Training DQN on: {device}")

    # Initialize environments
    env = setup_environment(config, seed=config.get("seed", 42))
    eval_env = setup_environment(config, seed=config.get("seed", 42))

    # Network architecture
    policy_kwargs = {
        "net_arch": [256, 128],
        "activation_fn": torch.nn.ReLU,
    }

    model = DQN(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/dqn/",
        seed=config.get("seed", 42),
    )

    # Callbacks - Simplified version
    callbacks = [
        MetricsCallback(),  # From base_train.py
        EvalCallback(
            eval_env,
            best_model_save_path="./models/dqn/best",
            log_path="./models/dqn/eval",
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            callback_on_new_best=MetricsCallback()  # Log metrics for best model
        )
    ]

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="dqn_run",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    
    model.save("./models/dqn/dqn_final")
    env.close()

if __name__ == "__main__":
    config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali",
        'grid_size': (5, 5),
        'layout': 'custom',
        'render_mode': None,
        'seed': 42,
    }
    train_dqn(config=config, total_timesteps=1_000_000)