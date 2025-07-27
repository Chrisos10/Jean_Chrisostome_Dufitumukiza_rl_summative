import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Tuple
from environment.rendering import StorageVisualizer
from environment.my_weather import get_weather

class StorageEnv(gym.Env):
    """
    Farm Storage Optimization Environment with Static State Zones
    
    Grid cells have permanent environmental state types (like Frozen Lake).
    Agent must navigate to optimal zones and perform zone-specific interventions
    based on current weather and crop conditions.
    """
    
    # Constants
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    
    # Static state types for grid cells
    STATE_TYPES = [
        {   # 0 - Optimal
            "name": "Optimal", 
            "color": (100, 255, 100),
            "ideal_temp": (20, 25), 
            "ideal_humidity": (60, 70),
            "actions": [0, 5],  # Do nothing, reduce quantity
            "action_effects": {0: (0, 0.5), 5: (-0.05, 1.0)}
        },
        {   # 1 - Cool Dry
            "name": "Cool Dry", 
            "color": (100, 200, 255),
            "ideal_temp": (15, 20), 
            "ideal_humidity": (50, 60),
            "actions": [1, 2],  # Ventilation controls
            "action_effects": {1: (-0.05, 1.5), 2: (0.1, -0.5)}
        },
        {   # 2 - Ventilated
            "name": "Ventilated", 
            "color": (200, 255, 200),
            "ideal_temp": (18, 28), 
            "ideal_humidity": (55, 65),
            "actions": [1, 3],  # Ventilation, natural repellent
            "action_effects": {1: (-0.1, 1.0), 3: (-0.15, 2.0)}
        },
        {   # 3 - Protected
            "name": "Protected", 
            "color": (255, 255, 100),
            "ideal_temp": (22, 26), 
            "ideal_humidity": (65, 75),
            "actions": [3, 4],  # Natural/diatomaceous treatments
            "action_effects": {3: (-0.2, 2.5), 4: (-0.3, 3.0)}
        },
        {   # 4 - Risk Zone
            "name": "Risk Zone", 
            "color": (255, 150, 100),
            "ideal_temp": None, 
            "ideal_humidity": None,
            "actions": [4, 6],  # Emergency actions
            "action_effects": {4: (-0.1, 1.0), 6: (0, 5.0)}
        }
    ]

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
            'grid_size': (5, 5),  # rows, columns
            'layout': 'cross'  # or 'random', 'rings'
        }
        
        self.render_mode = render_mode
        self.visualizer = None
        self.grid_size = self.config['grid_size']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]  # Start center
        
        # Initialize fixed grid states
        self._init_grid_states()
        
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
            "pest_level": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "position": spaces.Box(0, max(self.grid_size), shape=(2,), dtype=np.int32),
            "zone_type": spaces.Discrete(len(self.STATE_TYPES))
        })

        self.action_space = spaces.Discrete(11)

    def _init_grid_states(self):
        """Initialize the fixed grid state layout based on config"""
        self.grid_states = np.zeros(self.grid_size, dtype=int)
        
        if self.config.get('layout') == 'random':
            # Random distribution weighted by zone quality
            probs = [0.3, 0.25, 0.25, 0.15, 0.05]  # Sum to 1.0
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    self.grid_states[i,j] = np.random.choice(len(self.STATE_TYPES), p=probs)
        
        elif self.config.get('layout') == 'rings':
            # Concentric rings pattern
            center = (self.grid_size[0]//2, self.grid_size[1]//2)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    dist = max(abs(i-center[0]), abs(j-center[1]))
                    if dist == 0:
                        self.grid_states[i,j] = 0  # Optimal center
                    elif dist == 1:
                        self.grid_states[i,j] = random.choice([1, 2])  # Inner ring
                    elif dist == 2:
                        self.grid_states[i,j] = 3  # Outer ring
                    else:
                        self.grid_states[i,j] = 4  # Risk Zone edges
        
        else:  # Default 'cross' layout
            center = self.grid_size[0]//2
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if i == center or j == center:
                        # Central cross is optimal/ventilated
                        self.grid_states[i,j] = 0 if (i == center and j == center) else 2
                    else:
                        # Corners are risk zones
                        self.grid_states[i,j] = 4 if random.random() < 0.7 else 3

    def calculate_zone_suitability(self):
        """
        Calculate suitability score (0-1) for each zone based on:
        - Current temperature and humidity
        - Crop type preferences
        - Zone's ideal conditions
        """
        scores = np.zeros(self.grid_size)
        temp = self.state['temp'][0]
        humidity = self.state['humidity'][0]
        crop_type = self.state['crop_type']
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                zone = self.STATE_TYPES[self.grid_states[i,j]]
                
                if zone['name'] == "Risk Zone":
                    scores[i,j] = 0  # Always unsuitable
                    continue
                
                # Temperature match (0-1 where 1 is perfect)
                temp_mid = np.mean(zone['ideal_temp'])
                temp_score = 1 - min(1, abs(temp - temp_mid) / 15)
                
                # Humidity match with crop adjustments
                humidity_mid = np.mean(zone['ideal_humidity'])
                humidity_base = 1 - min(1, abs(humidity - humidity_mid) / 30)
                
                # Crop-specific adjustments
                if crop_type == 2:  # Rice prefers higher humidity
                    humidity_score = min(1, humidity_base * 1.3)
                elif crop_type == 0:  # Maize prefers moderate humidity
                    humidity_score = humidity_base * 0.9
                else:
                    humidity_score = humidity_base
                
                # Combined score with weights
                scores[i,j] = (temp_score * 0.6 + humidity_score * 0.4)
                
        return scores

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
        except Exception:
            temp, humidity = 28.0, 65.0

        # Initialize state
        self.state = {
            "temp": np.array([temp], dtype=np.float32),
            "humidity": np.array([humidity], dtype=np.float32),
            "crop_type": random.randint(0, 2),
            "storage_method": random.randint(0, 2),
            "duration": np.array([0], dtype=np.float32),
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)]
        }

        if self.render_mode == 'human' and self.visualizer:
            self.visualizer.reset()
            
        return self._flatten_state(self.state), {}

    def step(self, action):
        """Execute one environment step"""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        reward = 0
        terminated = False
        truncated = False
        self.last_action = action

        # Handle movement actions
        if action >= 7:  # Movement actions (7-10)
            new_pos = self._calculate_new_position(action)
            if new_pos != self.current_pos:
                reward += self._calculate_movement_reward(new_pos)
                self.current_pos = new_pos
        else:
            # Handle zone-specific actions
            reward += self._handle_zone_action(action)

        # Daily updates
        self._simulate_weather_change()
        self._update_pest_level()
        
        # Update state
        self.day += 1
        self.state.update({
            "duration": np.array([self.day], dtype=np.float32),
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)]
        })

        # Check termination conditions
        terminated = self._check_termination()
        
        info = {
            "current_action": self.ACTIONS[action],
            "day": self.day,
            "position": tuple(self.current_pos),
            "zone_type": self.STATE_TYPES[self.state["zone_type"]]["name"],
            "suitability": float(self.calculate_zone_suitability()[tuple(self.current_pos)]),
            "pest_level": float(self.pest_level)
        }

        return self._flatten_state(self.state), reward, terminated, truncated, info

    def _calculate_new_position(self, action):
        """Calculate new position after movement action"""
        new_pos = self.current_pos.copy()
        if action == 7:  # Up
            new_pos[0] = max(0, new_pos[0]-1)
        elif action == 8:  # Down
            new_pos[0] = min(self.grid_size[0]-1, new_pos[0]+1)
        elif action == 9:  # Left
            new_pos[1] = max(0, new_pos[1]-1)
        elif action == 10:  # Right
            new_pos[1] = min(self.grid_size[1]-1, new_pos[1]+1)
        return new_pos

    def _calculate_movement_reward(self, new_pos):
        """Calculate reward for moving to a new position"""
        target_zone = self.grid_states[tuple(new_pos)]
        suitability = self.calculate_zone_suitability()[tuple(new_pos)]
        
        # Zone type base rewards
        if target_zone == 4:  # Risk Zone
            return -2
        elif target_zone == 0:  # Optimal
            return 1.5 if suitability > 0.8 else 0.5
        
        # Suitability-based rewards
        if suitability > 0.7:
            return 1.2
        elif suitability < 0.3:
            return -0.5
        return 0.3

    def _handle_zone_action(self, action):
        """Handle non-movement actions with zone-specific effects"""
        zone_type = self.state["zone_type"]
        zone_info = self.STATE_TYPES[zone_type]
        
        # Check if action is valid for this zone
        if action not in zone_info["actions"]:
            return -1  # Penalty for invalid action
        
        # Apply zone-specific effects
        pest_change, action_reward = zone_info["action_effects"][action]
        self.pest_level = np.clip(self.pest_level + pest_change, 0, 1)
        
        # Additional effects for specific actions
        if action == 1:  # Increase ventilation
            self.state["temp"][0] = max(10, self.state["temp"][0] - 1)
            self.state["humidity"][0] = max(30, self.state["humidity"][0] - 2)
        elif action == 2:  # Decrease ventilation
            self.state["temp"][0] = min(40, self.state["temp"][0] + 1)
            self.state["humidity"][0] = min(90, self.state["humidity"][0] + 3)
            
        return action_reward

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
        """Calculate daily pest level changes"""
        base_risk = 0.05
        temp = self.state["temp"][0]
        humidity = self.state["humidity"][0]
        
        # Environmental effects
        temp_effect = max(0, temp - 25) * 0.01
        humidity_effect = max(0, humidity - 60) * 0.015
        
        # Storage method effect
        storage_risk = [0.0, 0.02, 0.05][self.state["storage_method"]]
        
        # Crop susceptibility
        crop_factor = [0.8, 1.0, 1.2][self.state["crop_type"]]
        
        # Zone risk modifier
        zone_risk = {
            0: 0.5,  # Optimal zones reduce risk
            1: 0.8,
            2: 0.7,
            3: 0.9,
            4: 1.5   # Risk zones increase danger
        }[self.state["zone_type"]]
        
        daily_increase = (base_risk + temp_effect + humidity_effect + storage_risk) * crop_factor * zone_risk
        self.pest_level = np.clip(self.pest_level + daily_increase, 0, 1)

    def _check_termination(self):
        """Check if episode should terminate"""
        return (
            self.pest_level >= 1.0 or  # Complete infestation
            self.day >= self.config['max_days'] or  # Season ended
            (self.last_action == 6)  # Harvest action
        )

    def _flatten_state(self, state):
        """Convert state dict to flat array for RL algorithms"""
        return np.concatenate([
            state["temp"],
            state["humidity"],
            [state["crop_type"]],
            [state["storage_method"]],
            state["duration"],
            state["pest_level"],
            state["position"],
            [state["zone_type"]]
        ]).astype(np.float32)

    def _init_visualization(self):
        """Initialize PyGame visualization"""
        try:
            self.visualizer = StorageVisualizer(self)
        except ImportError as e:
            print(f"Visualization disabled: {str(e)}")
            self.visualizer = None
            self.render_mode = 'console'

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
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def _console_render(self):
        """Text-based rendering for console"""
        print(f"\n=== Day {self.day}/{self.config['max_days']} ===")
        print(f"Position: {self.current_pos} | Zone: {self.STATE_TYPES[self.state['zone_type']]['name']}")
        print(f"Temp: {self.state['temp'][0]:.1f}°C | Humidity: {self.state['humidity'][0]:.1f}%")
        print(f"Crop: {self.CROP_TYPES[self.state['crop_type']]} | Storage: {self.STORAGE_METHODS[self.state['storage_method']]}")
        
        pest_bar = '█' * int(self.pest_level * 20) + '░' * (20 - int(self.pest_level * 20))
        print(f"Pest Risk: {pest_bar} {self.pest_level*100:.1f}%")
        
        if hasattr(self, 'last_action'):
            print(f"Last Action: {self.ACTIONS[self.last_action]}")
            print(f"Suitability: {self.calculate_zone_suitability()[tuple(self.current_pos)]:.2f}")

    def close(self):
        """Clean up resources"""
        if self.visualizer:
            self.visualizer.close()