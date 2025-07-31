import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C, PPO, DQN
from environment.custom_env import StorageEnv

class ModelEvaluator:
    """Comprehensive evaluation system for trained RL models"""
    
    def __init__(self, models_dir: str = "logs"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.results = {}
        
    def load_models(self) -> Dict:
        """Load all available trained models"""
        model_configs = {
            "A2C": ("a2c_run2/a2c_storage_model.zip", A2C),
            "A2C_v2": ("a2c_run/a2c_storage_model.zip", A2C), 
            "PPO": ("ppo_run/ppo_storage_model.zip", PPO),
            "DQN": ("dqn_run/dqn_storage_model.zip", DQN),
            "REINFORCE": ("reinforce_run/reinforce_storage_model.zip", A2C)
        }
        
        loaded_models = {}
        for name, (path, model_class) in model_configs.items():
            model_path = self.models_dir / path
            if model_path.exists():
                try:
                    model = model_class.load(str(model_path))
                    loaded_models[name] = model
                    print(f"✅ Loaded {name} model from {model_path}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        self.models = loaded_models
        return loaded_models
    
    def create_test_environment(self, config: Optional[Dict] = None) -> StorageEnv:
        """Create environment for testing"""
        default_config = {
            'max_days': 30,
            'initial_pest': 0.1,
            'location': "Kigali",
            'grid_size': (5, 5),
            'layout': 'custom',
            'obs_type': 'mlp',
            'curriculum_level': 1
        }
        
        if config:
            default_config.update(config)
            
        env = StorageEnv(config=default_config, obs_type="mlp", render_mode='console')
        return env
    
    def convert_action_to_int(self, action):
        """Robustly convert action to integer"""
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return int(action.item())
            else:
                return int(action.flatten()[0])
        else:
            return int(action)
    
    def evaluate_single_model(self, model_name: str, model, n_episodes: int = 10, 
                            render: bool = False, deterministic: bool = True) -> Dict:
        """Evaluate a single model comprehensively"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        env = self.create_test_environment()
        
        episode_data = []
        action_history = []
        zone_transitions = []
        pest_trajectories = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            episode_actions = []
            episode_zones = []
            episode_pest_levels = []
            prev_zone = None
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while not (done or truncated) and steps < 50:  # Limit steps to avoid infinite loops
                try:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    action = self.convert_action_to_int(action)
                    
                    episode_actions.append(action)
                    current_zone = info.get('zone_type', 'Unknown') if steps > 0 else env.STATE_TYPES[env.state['zone_type']]['name']
                    episode_zones.append(current_zone)
                    
                    if prev_zone is not None and prev_zone != current_zone:
                        zone_transitions.append((prev_zone, current_zone))
                    prev_zone = current_zone
                    
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    # Track pest levels
                    pest_level = info.get('pest_level', env.pest_level)
                    episode_pest_levels.append(pest_level)
                    
                    if render and steps % 5 == 0:
                        print(f"Step {steps}: Action={env.ACTIONS[action]}, Reward={reward:.2f}, "
                              f"Pest={pest_level:.3f}, Zone={current_zone}")
                        
                except Exception as e:
                    print(f"Error during step {steps}: {e}")
                    break
            
            episode_data.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': steps,
                'final_pest_level': episode_pest_levels[-1] if episode_pest_levels else 1.0,
                'success': episode_pest_levels[-1] < 0.8 if episode_pest_levels else False,
                'actions_taken': len(episode_actions),
                'unique_actions': len(set(episode_actions)),
                'zones_visited': len(set(episode_zones))
            })
            
            action_history.extend(episode_actions)
            pest_trajectories.append(episode_pest_levels)
            
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}, "
                  f"Final Pest={episode_pest_levels[-1] if episode_pest_levels else 'N/A':.3f}")
        
        env.close()
        
        # Calculate comprehensive metrics
        results = self._calculate_metrics(episode_data, action_history, zone_transitions, 
                                        pest_trajectories, env.ACTIONS)
        results['model_name'] = model_name
        
        return results
    
    def _calculate_metrics(self, episode_data: List[Dict], action_history: List[int], 
                          zone_transitions: List[Tuple], pest_trajectories: List[List[float]],
                          action_names: List[str]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        df = pd.DataFrame(episode_data)
        
        # Basic performance metrics
        metrics = {
            'mean_reward': df['total_reward'].mean(),
            'std_reward': df['total_reward'].std(),
            'mean_steps': df['steps'].mean(),
            'mean_final_pest': df['final_pest_level'].mean(),
            'success_rate': df['success'].mean() * 100,
            'completion_rate': (df['steps'] < 50).mean() * 100,  # Episodes that didn't hit step limit
        }
        
        # Action analysis
        if action_history:
            action_counts = np.bincount(action_history, minlength=len(action_names))
            action_distribution = action_counts / len(action_history) * 100
            
            top_actions = sorted(enumerate(action_distribution), key=lambda x: x[1], reverse=True)[:5]
            metrics['top_actions'] = [(action_names[idx], pct) for idx, pct in top_actions]
            metrics['action_diversity'] = len([x for x in action_counts if x > 0]) / len(action_names)
        
        # Pest control effectiveness
        if pest_trajectories:
            all_pest_improvements = []
            for trajectory in pest_trajectories:
                if len(trajectory) > 1:
                    improvement = trajectory[0] - trajectory[-1]  # Positive = improvement
                    all_pest_improvements.append(improvement)
            
            if all_pest_improvements:
                metrics['mean_pest_improvement'] = np.mean(all_pest_improvements)
                metrics['pest_improvement_success_rate'] = sum(1 for x in all_pest_improvements if x > 0) / len(all_pest_improvements) * 100
        
        # Zone transition analysis
        if zone_transitions:
            transition_counts = {}
            for from_zone, to_zone in zone_transitions:
                key = f"{from_zone} -> {to_zone}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            most_common_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            metrics['common_transitions'] = most_common_transitions
        
        metrics['episode_data'] = episode_data
        metrics['action_history'] = action_history
        metrics['pest_trajectories'] = pest_trajectories
        
        return metrics
    
    def compare_models(self, n_episodes: int = 10, save_results: bool = True) -> pd.DataFrame:
        """Compare all loaded models"""
        print("\n🏆 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 50)
        
        if not self.models:
            print("No models loaded. Please run load_models() first.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model in self.models.items():
            try:
                results = self.evaluate_single_model(model_name, model, n_episodes, render=False)
                self.results[model_name] = results
                
                comparison_data.append({
                    'Model': model_name,
                    'Mean Reward': results['mean_reward'],
                    'Std Reward': results['std_reward'],
                    'Success Rate (%)': results['success_rate'],
                    'Mean Steps': results['mean_steps'],
                    'Final Pest Level': results['mean_final_pest'],
                    'Action Diversity': results.get('action_diversity', 0),
                    'Pest Improvement': results.get('mean_pest_improvement', 0)
                })
                
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {e}")
                
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Sort by success rate, then by mean reward
            comparison_df = comparison_df.sort_values(['Success Rate (%)', 'Mean Reward'], 
                                                    ascending=[False, False])
            
            print("\n📊 MODEL COMPARISON RESULTS")
            print(comparison_df.round(3).to_string(index=False))
            
            # Identify best model
            best_model = comparison_df.iloc[0]['Model']
            print(f"\n🥇 BEST PERFORMING MODEL: {best_model}")
            
            if save_results:
                self._save_comparison_results(comparison_df)
                self._create_visualizations()
        
        return comparison_df
    
    def _save_comparison_results(self, comparison_df: pd.DataFrame):
        """Save comparison results to files"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison table
        comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
        
        # Save detailed results
        with open(results_dir / "detailed_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, results in self.results.items():
                serializable_results[model_name] = {}
                for key, value in results.items():
                    if isinstance(value, (np.ndarray, list)):
                        if key == 'pest_trajectories':
                            serializable_results[model_name][key] = [
                                [float(x) for x in traj] for traj in value
                            ]
                        else:
                            serializable_results[model_name][key] = [
                                float(x) if isinstance(x, (np.float32, np.float64)) else x 
                                for x in (value if isinstance(value, list) else value.tolist())
                            ]
                    else:
                        serializable_results[model_name][key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Results saved to {results_dir}")
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            return
            
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        rewards = [self.results[m]['mean_reward'] for m in models]
        errors = [self.results[m]['std_reward'] for m in models]
        
        bars = ax1.bar(models, rewards, yerr=errors, capsize=5, alpha=0.7)
        ax1.set_title('Mean Episode Reward by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # 2. Success Rate Comparison
        ax2 = plt.subplot(2, 3, 2)
        success_rates = [self.results[m]['success_rate'] for m in models]
        bars = ax2.bar(models, success_rates, alpha=0.7, color='green')
        ax2.set_title('Success Rate (Pest Level < 0.8)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Pest Control Effectiveness
        ax3 = plt.subplot(2, 3, 3)
        final_pest_levels = [self.results[m]['mean_final_pest'] for m in models]
        bars = ax3.bar(models, final_pest_levels, alpha=0.7, color='red')
        ax3.set_title('Mean Final Pest Level', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Final Pest Level')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0.8, color='orange', linestyle='--', label='Failure Threshold')
        ax3.legend()
        
        # 4. Action Diversity
        ax4 = plt.subplot(2, 3, 4)
        diversities = [self.results[m].get('action_diversity', 0) for m in models]
        bars = ax4.bar(models, diversities, alpha=0.7, color='blue')
        ax4.set_title('Action Diversity (Fraction of Actions Used)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Action Diversity')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Episode Length Distribution
        ax5 = plt.subplot(2, 3, 5)
        for model_name in models:
            episode_data = self.results[model_name].get('episode_data', [])
            if episode_data:
                steps = [ep['steps'] for ep in episode_data]
                ax5.hist(steps, alpha=0.6, label=model_name, bins=10)
        
        ax5.set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Steps per Episode')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # 6. Pest Level Trajectories
        ax6 = plt.subplot(2, 3, 6)
        for model_name in models:
            trajectories = self.results[model_name].get('pest_trajectories', [])
            if trajectories:
                # Average trajectory
                max_len = max(len(traj) for traj in trajectories)
                padded_trajectories = []
                for traj in trajectories:
                    padded = traj + [traj[-1]] * (max_len - len(traj))  # Pad with last value
                    padded_trajectories.append(padded)
                
                mean_trajectory = np.mean(padded_trajectories, axis=0)
                std_trajectory = np.std(padded_trajectories, axis=0)
                steps = range(len(mean_trajectory))
                
                ax6.plot(steps, mean_trajectory, label=model_name, marker='o', markersize=3)
                ax6.fill_between(steps, 
                               mean_trajectory - std_trajectory, 
                               mean_trajectory + std_trajectory, 
                               alpha=0.2)
        
        ax6.set_title('Average Pest Level Trajectory', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Pest Level')
        ax6.axhline(y=0.8, color='red', linestyle='--', label='Failure Threshold')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "model_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {results_dir}/model_comparison_plots.png")
    
    def interactive_demo(self, model_name: str, n_episodes: int = 3):
        """Run an interactive demonstration of a specific model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return
        
        print(f"\n🎮 INTERACTIVE DEMO: {model_name}")
        print("=" * 50)
        
        model = self.models[model_name]
        env = self.create_test_environment()
        
        for episode in range(n_episodes):
            print(f"\n🎯 EPISODE {episode + 1}")
            print("-" * 30)
            
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            # Print initial state
            env.render()
            input("Press Enter to start the episode...")
            
            while not (done or truncated) and steps < 30:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    action = self.convert_action_to_int(action)
                    
                    print(f"\n🤖 Model chooses: {env.ACTIONS[action]} (Action {action})")
                    
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    env.render()
                    
                    print(f"💰 Reward: {reward:.2f} | Total: {total_reward:.2f}")
                    
                    if done or truncated:
                        print(f"\n🏁 Episode finished!")
                        print(f"   Final reward: {total_reward:.2f}")
                        print(f"   Final pest level: {info.get('pest_level', 'N/A'):.3f}")
                        print(f"   Steps taken: {steps}")
                        success = info.get('pest_level', 1.0) < 0.8
                        print(f"   Success: {'✅ YES' if success else '❌ NO'}")
                        break
                    
                    input("Press Enter for next step...")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    break
        
        env.close()

def main():
    """Main execution function"""
    print("🚀 RL Model Evaluation System")
    print("=" * 40)
    
    evaluator = ModelEvaluator()
    
    # Load all available models
    loaded_models = evaluator.load_models()
    
    if not loaded_models:
        print("❌ No models found! Please train models first.")
        return
    
    print(f"\n📋 Found {len(loaded_models)} models: {list(loaded_models.keys())}")
    
    # Run comprehensive comparison
    print("\n🔄 Running comprehensive evaluation...")
    comparison_df = evaluator.compare_models(n_episodes=15, save_results=True)
    
    if not comparison_df.empty:
        print("\n🎉 Evaluation complete!")
        
        # Ask user if they want an interactive demo
        best_model = comparison_df.iloc[0]['Model']
        response = input(f"\nWould you like to see an interactive demo of the best model ({best_model})? (y/n): ")
        
        if response.lower() == 'y':
            evaluator.interactive_demo(best_model, n_episodes=2)
        
        # Ask about other models
        print(f"\nAvailable models for demo: {list(loaded_models.keys())}")
        model_choice = input("Enter model name for interactive demo (or press Enter to skip): ")
        
        if model_choice and model_choice in loaded_models:
            evaluator.interactive_demo(model_choice, n_episodes=2)

if __name__ == "__main__":
    main()