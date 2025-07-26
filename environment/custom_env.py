import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.rendering import StorageVisualizer
from typing import Optional, Dict, Tuple
from environment.my_weather import get_weather

class StorageEnv(gym.Env):
    """Enhanced Farm Storage Optimization Environment with spatial grid system"""
    
    # Constants
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    ACTIONS = [
        "Do nothing",
        "Increase ventilation",
        "Decrease ventilation",
        "Apply natural repellent",
        "Apply diatomaceous earth",
        "Reduce quantity",
        "Immediate harvest/sell",
        "Move up",
        "Move down",
        "Move left",
        "Move right"
    ]
    
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        # Configuration with defaults
        self.config = config or {
            'max_days': 30,
            'initial_pest': 0.1,
            'location': "Kigali",
            'risk_factors': {
                'temp_high_risk': 35,
                'humidity_high_risk': 70,
                'base_risk': 0.05
            },
            'grid_size': (5, 5)  # rows, columns
        }
        
        self.render_mode = render_mode
        self.visualizer = None
        self.grid_size = self.config['grid_size']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]  # Center position
        
        # Initialize visualization
        if self.render_mode == 'human':
            self._init_visualization()

        # Enhanced observation space
        self.observation_space = spaces.Dict({
            "temp": spaces.Box(0, 50, shape=(1,), dtype=np.float32),
            "humidity": spaces.Box(0, 100, shape=(1,), dtype=np.float32),
            "crop_type": spaces.Discrete(3),
            "storage_method": spaces.Discrete(3),
            "duration": spaces.Box(0, self.config['max_days'], shape=(1,), dtype=np.float32),
            "previous_pest": spaces.Discrete(2),
            "pest_level": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "position": spaces.Box(0, max(self.grid_size), shape=(2,), dtype=np.int32)
        })

        # Expanded action space
        self.action_space = spaces.Discrete(11)  # 7 original + 4 movement actions

    def _init_visualization(self):
        """Initialize PyGame visualization"""
        try:
            self.visualizer = StorageVisualizer(self)
        except ImportError as e:
            print(f"Visualization disabled: {str(e)}")
            self.visualizer = None
            self.render_mode = 'console'

    def _get_zone_risk(self, pos):
        """Calculate risk for grid position"""
        # Higher risk near edges
        row_risk = abs(pos[0] - self.grid_size[0]/2) / (self.grid_size[0]/2)
        col_risk = abs(pos[1] - self.grid_size[1]/2) / (self.grid_size[1]/2)
        return (row_risk + col_risk) / 2

    def _get_recommended_action(self):
        """Get recommended action based on current state"""
        pest = self.pest_level
        temp = self.state['temp'][0]
        humidity = self.state['humidity'][0]
        
        if pest > 0.7:
            return 4  # Diatomaceous earth
        elif pest > 0.4:
            return 3  # Natural repellent
        elif temp > 30 or humidity > 75:
            return 1  # Increase ventilation
        else:
            return 0  # Do nothing

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.day = 0
        self.pest_level = self.config['initial_pest']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]
        self.last_action = None
        
        # Get weather data
        try:
            temp, humidity = get_weather(self.config['location'])
        except Exception as e:
            print(f"Weather error: {str(e)}. Using defaults.")
            temp, humidity = 28.0, 65.0

        # Initialize state
        self.state = {
            "temp": np.array([temp], dtype=np.float32),
            "humidity": np.array([humidity], dtype=np.float32),
            "crop_type": random.randint(0, 2),
            "storage_method": random.randint(0, 2),
            "duration": np.array([0], dtype=np.float32),
            "previous_pest": 0,
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32)
        }

        if self.render_mode == 'human' and self.visualizer:
            self.visualizer.reset()
            
        return self._flatten_state(self.state), {}

    def step(self, action):
        """Execute one environment step"""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        reward = 0.0
        terminated = False
        truncated = False
        self.last_action = action

        # Handle movement actions
        if action == 7:  # Up
            self.current_pos[0] = max(0, self.current_pos[0]-1)
        elif action == 8:  # Down
            self.current_pos[0] = min(self.grid_size[0]-1, self.current_pos[0]+1)
        elif action == 9:  # Left
            self.current_pos[1] = max(0, self.current_pos[1]-1)
        elif action == 10:  # Right
            self.current_pos[1] = min(self.grid_size[1]-1, self.current_pos[1]+1)
        else:
            # Original action effects
            pest_change, action_reward = {
                0: (0, -1),      # Do nothing
                1: (-0.05, 1),   # Increase ventilation
                2: (0.1, -1),    # Decrease ventilation
                3: (-0.1, 2),    # Natural repellent
                4: (-0.15, 3),   # Diatomaceous earth
                5: (-0.05, 1),   # Reduce quantity
                6: (0, 5 if self.pest_level < 0.5 else -5)  # Harvest/sell
            }[action]
            self.pest_level = np.clip(self.pest_level + pest_change, 0, 1)
            reward += action_reward

        # Daily updates
        self._simulate_weather_change()
        self._update_pest_level()
        
        # Position affects pest growth
        zone_risk = self._get_zone_risk(self.current_pos)
        self.pest_level = np.clip(self.pest_level + zone_risk*0.02, 0, 1)
        
        # Update state
        self.day += 1
        self.state.update({
            "duration": np.array([self.day], dtype=np.float32),
            "previous_pest": 1 if self.pest_level > 0.3 else 0,
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32)
        })

        # Reward shaping
        recommended = self._get_recommended_action()
        if action == recommended:
            reward += 2  # Bonus for following recommendation
            
        if self.pest_level > 0.8:
            reward -= 5  # High risk penalty
        if action in [3, 4] and self.pest_level > 0.3:
            reward *= 1.5  # Bonus for timely treatment
            
        terminated = self._check_termination()
        
        self.last_info = {
            "current_action": self.ACTIONS[action],  # Changed from "action"
            "recommended": self.ACTIONS[recommended],
            "day": self.day,
            "position": tuple(self.current_pos),
            "pest_level": float(self.pest_level),
            "weather": (float(self.state["temp"][0]), float(self.state["humidity"][0]))
        }

        return self._flatten_state(self.state), reward, terminated, truncated, self.last_info

    def _flatten_state(self, state):
        """Convert state dict to flat array"""
        return np.concatenate([
            state["temp"],
            state["humidity"],
            [state["crop_type"]],
            [state["storage_method"]],
            state["duration"],
            [state["previous_pest"]],
            state["pest_level"],
            state["position"]
        ])

    def _simulate_weather_change(self):
        """Simulate daily weather fluctuations"""
        temp_change = random.uniform(-2, 2)
        humidity_change = random.uniform(-5, 5)
        
        self.state["temp"] = np.clip(
            self.state["temp"] + temp_change,
            10, 40  # realistic temp range
        )
        self.state["humidity"] = np.clip(
            self.state["humidity"] + humidity_change,
            30, 90  # realistic humidity range
        )

    def _update_pest_level(self):
        """Calculate pest level increase based on conditions"""
        base_risk = self.config['risk_factors']['base_risk']
        
        # Environmental effects
        temp_effect = max(0, self.state["temp"][0] - 25) * 0.01
        humidity_effect = max(0, self.state["humidity"][0] - 60) * 0.015
        storage_effect = [0.0, 0.02, 0.05][self.state["storage_method"]]
        crop_factor = [0.8, 1.0, 1.2][self.state["crop_type"]]
        
        risk_increase = (base_risk + temp_effect + humidity_effect + storage_effect) * crop_factor
        self.pest_level = np.clip(self.pest_level + risk_increase, 0, 1)

    def _check_termination(self):
        """Check if episode should terminate"""
        return (
            self.pest_level >= 1.0 or 
            self.day >= self.config['max_days'] or 
            (self.last_action == 6)  # Harvest action
        )

    def _console_render(self):
        """Enhanced text-based rendering"""
        print(f"\n=== Day {self.day}/{self.config['max_days']} ===")
        print(f"Position: {self.current_pos}")
        print(f"Temp: {self.state['temp'][0]:.1f}°C | Humidity: {self.state['humidity'][0]:.1f}%")
        print(f"Crop: {self.CROP_TYPES[self.state['crop_type']]} | Storage: {self.STORAGE_METHODS[self.state['storage_method']]}")
        
        pest_bar = '█' * int(self.pest_level * 20) + '░' * (20 - int(self.pest_level * 20))
        print(f"Pest Risk: {pest_bar} {self.pest_level*100:.1f}%")
        
        if hasattr(self, 'last_info'):
            print(f"Current Action: {self.last_info['current_action']}")
            print(f"Recommended: {self.last_info['recommended']}")

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human' and self.visualizer:
            return self.visualizer.render()
        elif self.render_mode == 'console' or self.render_mode is None:
            self._console_render()
            return None
        elif self.render_mode == 'rgb_array':
            if self.visualizer:
                return self.visualizer.get_rgb_array()
            else:
                return np.zeros((600, 800, 3), dtype=np.uint8)
        return None
        
    def close(self):
        """Clean up resources"""
        if self.visualizer:
            self.visualizer.close()