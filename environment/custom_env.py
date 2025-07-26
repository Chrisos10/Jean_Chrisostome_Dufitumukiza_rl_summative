import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.rendering import StorageVisualizer
from typing import Optional, Dict, Tuple
from environment.my_weather import get_weather  # Your weather module (real or mock)

class StorageEnv(gym.Env):
    """
    Final Farm Storage Optimization Environment
    with optional PyGame visualization support
    """
    
    # Constants for readability
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    ACTIONS = [
        "Do nothing",
        "Increase ventilation",
        "Decrease ventilation",
        "Apply natural repellent",
        "Apply diatomaceous earth",
        "Reduce quantity",
        "Immediate harvest/sell"
    ]
    
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None):
        """
        Args:
            config: Environment configuration dictionary
            render_mode: None, 'human', 'rgb_array', or 'console'
        """
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
            }
        }
        
        self.render_mode = render_mode
        self.visualizer = None
        
        # Initialize visualization if requested
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
            "pest_level": spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        })

        # Action space
        self.action_space = spaces.Discrete(7)

        # Environment state
        self.day = 0
        self.pest_level = self.config['initial_pest']
        self.state = None
        self.last_action = None

    def _init_visualization(self):
        """Initialize PyGame visualization if available"""
        try:
            self.visualizer = StorageVisualizer(self)
        except ImportError as e:
            print(f"Visualization disabled: {str(e)}")
            self.visualizer = None
            self.render_mode = 'console'

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.day = 0
        self.pest_level = self.config['initial_pest']
        self.last_action = None

        # Get weather data
        try:
            temperature, humidity = get_weather(location=self.config['location'])
        except Exception as e:
            print(f"Weather API error: {str(e)}. Using defaults.")
            temperature, humidity = 28.0, 65.0  # Fallback values

        # Initialize state
        self.state = {
            "temp": np.array([temperature], dtype=np.float32),
            "humidity": np.array([humidity], dtype=np.float32),
            "crop_type": random.randint(0, 2),
            "storage_method": random.randint(0, 2),
            "duration": np.array([0], dtype=np.float32),
            "previous_pest": 0,
            "pest_level": np.array([self.pest_level], dtype=np.float32)
        }

        if self.render_mode == 'human' and self.visualizer:
            self.visualizer.reset()

        return self._flatten_state(self.state), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        self.last_action = action
        reward = 0.0
        terminated = False
        truncated = False

        # Simulate daily changes
        self._simulate_weather_change()
        self._update_pest_level()
        
        # Apply action effects
        reward = self._apply_action_effects(action)
        
        # Update state
        self.day += 1
        self.state["duration"] = np.array([self.day], dtype=np.float32)
        self.state["previous_pest"] = 1 if self.pest_level > 0.3 else 0
        self.state["pest_level"] = np.array([self.pest_level], dtype=np.float32)
        
        # Check termination
        terminated = self._check_termination()
        
        info = {
            "action": self.ACTIONS[action],
            "day": self.day,
            "pest_level": float(self.pest_level),
            "weather": (float(self.state["temp"][0]), float(self.state["humidity"][0]))
        }

        return self._flatten_state(self.state), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
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
                return self._get_rgb_array()
        return None
        
    def close(self):
        """Clean up resources"""
        if self.visualizer:
            self.visualizer.close()

    # ===== Helper Methods =====
    def _flatten_state(self, state: Dict) -> np.ndarray:
        """Convert state dict to flat array for compatibility"""
        return np.concatenate([
            state["temp"],
            state["humidity"],
            [state["crop_type"]],
            [state["storage_method"]],
            state["duration"],
            [state["previous_pest"]],
            state["pest_level"]
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

    def _apply_action_effects(self, action: int) -> float:
        """Apply action effects and return reward"""
        action_effects = {
            0: (0, -1),       # Do nothing
            1: (-0.05, 1),    # Increase ventilation
            2: (0.1, -1),     # Decrease ventilation
            3: (-0.1, 2),     # Natural repellent
            4: (-0.15, 3),    # Diatomaceous earth
            5: (-0.05, 1),    # Reduce quantity
            6: (0, 5 if self.pest_level < 0.5 else -5)  # Harvest/sell
        }
        
        pest_change, reward = action_effects[action]
        self.pest_level = np.clip(self.pest_level + pest_change, 0, 1)
        
        # Additional reward shaping
        if self.pest_level > 0.8:
            reward -= 5  # High risk penalty
        if action in [3, 4] and self.pest_level > 0.3:
            reward *= 1.5  # Bonus for timely treatment
            
        return reward

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        return (
            self.pest_level >= 1.0 or 
            self.day >= self.config['max_days'] or 
            (self.last_action == 6)  # Harvest action
        )

    def _console_render(self):
        """Text-based rendering for console"""
        print(f"\n=== Day {self.day}/{self.config['max_days']} ===")
        print(f"Temp: {self.state['temp'][0]:.1f}°C | Humidity: {self.state['humidity'][0]:.1f}%")
        print(f"Crop: {self.CROP_TYPES[self.state['crop_type']]} | Storage: {self.STORAGE_METHODS[self.state['storage_method']]}")
        
        pest_bar = '█' * int(self.pest_level * 20) + '░' * (20 - int(self.pest_level * 20))
        print(f"Pest Risk: {pest_bar} {self.pest_level*100:.1f}%")
        
        if self.last_action is not None:
            print(f"Last Action: {self.ACTIONS[self.last_action]}")

    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array for video recording (fallback)"""
        if self.visualizer:
            return self.visualizer.get_rgb_array()
        return np.zeros((600, 800, 3), dtype=np.uint8)