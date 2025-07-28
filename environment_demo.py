import numpy as np
from environment.custom_env import StorageEnv
import imageio
import os
from datetime import datetime

def create_demo_gif(num_steps=50, output_dir="demos"):
    """Create a GIF of random agent actions"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize environment with visualization
    config = {
        'max_days': 30,
        'initial_pest': 0.1,
        'location': "Kigali",
        'grid_size': (5, 5),
        'layout': 'custom'
    }
    env = StorageEnv(config=config, render_mode='human')
    
    # Reset environment
    obs, _ = env.reset()
    frames = []
    
    # Run random actions
    for _ in range(num_steps):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Reset if episode ends
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Save GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"random_agent_{timestamp}.gif")
    imageio.mimsave(gif_path, frames, fps=5)
    print(f"Saved demo GIF to: {gif_path}")
    
    env.close()

if __name__ == "__main__":
    create_demo_gif(num_steps=100)