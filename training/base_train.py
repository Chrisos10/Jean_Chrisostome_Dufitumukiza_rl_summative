import os
import random
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.custom_env import StorageEnv

class MetricsCallback(BaseCallback):
    """Logs custom environment metrics to TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.risk_zone_time = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # Log every 100 steps
            env = self.training_env.envs[0].env
            zone_type = env.state["zone_type"]
            
            # Track risk zone time
            if zone_type == 4:
                self.risk_zone_time += 1
            else:
                self.risk_zone_time = max(0, self.risk_zone_time - 0.5)
            
            # Log metrics
            self.logger.record("metrics/pest_level", env.pest_level)
            self.logger.record("metrics/time_in_risk", self.risk_zone_time)
            self.logger.record("metrics/zone_suitability", 
                              env.calculate_zone_suitability()[tuple(env.current_pos)])
        return True

class CustomEvalCallback(EvalCallback):
    """Extended evaluation with custom metrics"""
    def _on_step(self) -> bool:
        result = super()._on_step()
        if result and self.n_calls % self.eval_freq == 0:
            # Calculate average metrics across evaluation episodes
            pest_levels = [info.get("pest_level", 0) for info in self.eval_infos]
            risk_times = [info.get("time_in_risk", 0) for info in self.eval_infos]
            
            self.logger.record("eval/avg_pest_level", np.mean(pest_levels))
            self.logger.record("eval/max_pest_level", np.max(pest_levels))
            self.logger.record("eval/avg_risk_time", np.mean(risk_times))
        return result

def setup_environment(config, seed=None):
    """Create and wrap the environment with seed control"""
    def _init_env():
        env = StorageEnv(config=config, render_mode=None)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    env = DummyVecEnv([_init_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        norm_obs_keys=["temp", "humidity", "duration", "pest_level"],
        training=True
    )
    return env

def set_global_seeds(seed):
    """Set seeds for reproducibility across all libraries"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gym.utils.seeding.np_random(seed)

def get_device():
    """Check for CUDA availability with fallback to CPU"""
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).cuda()
            return torch.device("cuda")
        except RuntimeError:
            print("CUDA device found but not usable, falling back to CPU")
    return torch.device("cpu")

def create_callbacks(log_dir, eval_env, save_freq=10000, eval_freq=5000):
    """Create comprehensive callback suite"""
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)
    
    return CallbackList([
        MetricsCallback(),  # Training metrics
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True
        ),
        CustomEvalCallback(  # Enhanced evaluation
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
    ])