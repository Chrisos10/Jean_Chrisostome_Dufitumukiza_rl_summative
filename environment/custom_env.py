import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Tuple
from environment.rendering import StorageVisualizer
from environment.my_weather import get_weather
import cv2

class StorageEnv(gym.Env):
    """Enhanced Farm Storage Environment with improved learning dynamics"""

    # Constants
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    
    STATE_TYPES = [
        {   # 0 - Optimal Storage
            "name": "Optimal Storage", 
            "color": (100, 255, 100),
            "ideal_temp": (18, 25), 
            "ideal_humidity": (60, 70),
            "description": "Perfect balanced conditions - maintain with minimal intervention",
            "actions": [0, 6, 7],  # Monitor, temperature control, preventive neem
            "action_effects": {
                0: (-0.005, 0.8),    # Just monitoring - very stable conditions
                6: (-0.01, 0.6),     # Fine-tune temperature to maintain perfection
                7: (-0.015, 1.0)     # Preventive neem application
            }
        },
        {   # 1 - Too Dry (Low humidity, may cause grain brittleness)
            "name": "Too Dry", 
            "color": (100, 200, 255),
            "ideal_temp": (15, 22), 
            "ideal_humidity": (35, 50),
            "description": "Dangerously low humidity - grain may crack, dust increases pest attraction",
            "actions": [2, 4, 8, 12],  # Reduce ventilation, add controlled moisture, moisture control, sealed containers
            "action_effects": {
                2: (-0.02, 0.7),     # Reduce ventilation to retain moisture
                4: (-0.04, 1.2),     # Add controlled moisture - critical here
                8: (-0.03, 0.9),     # General moisture management
                12: (-0.05, 1.1)     # Sealed containers prevent further moisture loss
            }
        },
        {   # 2 - Well Ventilated
            "name": "Well Ventilated", 
            "color": (200, 255, 200),
            "ideal_temp": (18, 28), 
            "ideal_humidity": (55, 65),
            "description": "Excellent airflow - perfect for preventing mold and fungal issues",
            "actions": [0, 1, 5, 10],  # Monitor, increase ventilation, solar drying, aromatic herbs
            "action_effects": {
                0: (-0.01, 0.5),     # Natural air circulation is working well
                1: (-0.08, 1.3),     # Increase ventilation - very effective here
                5: (-0.06, 1.0),     # Solar drying complements ventilation
                10: (-0.07, 1.1)     # Aromatic herbs work excellently with air circulation
            }
        },
        {   # 3 - High Humidity
            "name": "High Humidity", 
            "color": (255, 255, 100),
            "ideal_temp": (22, 30), 
            "ideal_humidity": (75, 90),
            "description": "Dangerous moisture levels - mold and fungal growth imminent",
            "actions": [1, 3, 5, 9, 11],  # Increase ventilation, moisture absorbers, solar drying, ash, DE
            "action_effects": {
                1: (-0.10, 1.5),     # Ventilation critical for humidity
                3: (-0.12, 1.8),     # Moisture absorbers essential
                5: (-0.15, 2.0),     # Solar drying very effective
                9: (-0.08, 1.2),     # Wood ash absorbs moisture and deters pests
                11: (-0.18, 2.2)     # DE works well in humid conditions
            }
        },
        {   # 4 - Critical Risk Zone
            "name": "Critical Risk", 
            "color": (255, 50, 50),
            "ideal_temp": None, 
            "ideal_humidity": None,
            "description": "EMERGENCY: Active pest infestation or severe environmental damage detected",
            "actions": [11, 13, 14, 15, 16],  # Emergency protocols only
            "action_effects": {
                11: (-0.25, 2.0),    # Diatomaceous earth - immediate pest control
                13: (-0.35, 3.0),    # Multi-method emergency treatment
                14: (-0.20, 2.5),    # Remove infected portions immediately
                15: (-0.05, 5.0),    # Emergency harvest to save what's left
                16: (-0.40, 3.5)     # Full integrated emergency response
            }
        }
    ]

    ACTIONS = [
        "Monitor and maintain",          # 0 - Passive monitoring with basic maintenance
        "Increase ventilation",          # 1 - Open vents, fans, improve airflow
        "Reduce ventilation",            # 2 - Close vents to retain/control moisture
        "Add moisture absorbers",        # 3 - Silica gel, lime, dry materials
        "Add controlled moisture",       # 4 - Slight humidification if too dry
        "Solar drying",                  # 5 - Use solar energy to reduce moisture
        "Temperature regulation",        # 6 - Shade, insulation, thermal mass
        "Apply neem treatment",          # 7 - Neem oil/leaves - natural pesticide
        "Moisture level control",        # 8 - Comprehensive moisture management
        "Apply wood ash",                # 9 - Traditional ash treatment for pests/moisture
        "Use aromatic herbs",            # 10 - Mint, basil, eucalyptus leaves
        "Apply diatomaceous earth",      # 11 - Food-grade DE for pest control
        "Use sealed containers",         # 12 - Airtight storage for portions
        "Intensive natural treatment",   # 13 - Combined natural methods
        "Remove affected portions",      # 14 - Isolate damaged grain
        "Emergency harvest",             # 15 - Immediate sale/use of crop
        "Integrated pest management",    # 16 - Multiple natural methods combined
        "Move up",                       # 17
        "Move down",                     # 18
        "Move left",                     # 19
        "Move right"                     # 20
    ]
    
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None, obs_type: str = "multi"):
        super().__init__()
        
        self.config = config or {
            'max_days': 30,
            'initial_pest': 0.1,
            'location': "Kigali",
            'grid_size': (5, 5),
            'layout': 'custom',
            'obs_type': 'multi'
        }
        
        self.obs_type = self.config.get('obs_type', obs_type)
        self.render_mode = render_mode
        self.visualizer = None
        self.grid_size = self.config['grid_size']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]
        self.time_in_risk_zone = 0
        self.visited_positions = set()
        self.last_action = None
        self.step_count = 0
        
        # Learning enhancement variables
        self.cumulative_reward = 0
        self.best_pest_level = 1.0
        self.exploration_bonus = 0
        
        self._init_grid_states()
        if self.render_mode == 'human':
            self._init_visualization()

        # Updated observation spaces to account for new action count
        if self.obs_type == "mlp":
            self.observation_space = spaces.Box(
                low=-10, high=100, 
                shape=(12,),
                dtype=np.float32
            )
        elif self.obs_type == "cnn":
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3),
                dtype=np.uint8
            )
        else:  # multi
            self.observation_space = spaces.Dict({
                "vector": spaces.Box(low=-10, high=100, shape=(12,), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            })

        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def _init_grid_states(self):
        """Initialize grid with dynamic risk zone placement based on training phase"""
        self.grid_states = np.zeros(self.grid_size, dtype=int)
        
        # Get curriculum level (0=easy, 1=medium, 2=hard, 3=expert)
        curriculum_level = self.config.get('curriculum_level', 1)
        
        if self.config.get('layout') == 'custom':
            if curriculum_level == 0:  # Easy: Mostly optimal and ventilated zones
                custom_pattern = [
                    [2, 0, 3, 2, 0],
                    [0, 3, 2, 0, 2],
                    [3, 2, 0, 3, 2],
                    [2, 0, 2, 0, 1],
                    [1, 3, 4, 2, 4]  # Two risk zones to test
                ]
            elif curriculum_level == 1:  # Medium: Mix of conditions with some challenges
                custom_pattern = [
                    [4, 2, 1, 3, 2],
                    [0, 3, 4, 1, 2],
                    [3, 2, 0, 3, 4],
                    [2, 4, 2, 0, 1],
                    [1, 3, 4, 2, 3]
                ]
            elif curriculum_level == 2:  # Hard: More challenging conditions
                custom_pattern = [
                    [4, 2, 4, 3, 2],
                    [1, 4, 2, 4, 3],
                    [4, 3, 0, 3, 4],
                    [2, 4, 3, 4, 1],
                    [4, 3, 4, 2, 4]
                ]
            else:  # Expert: Maximum challenge
                custom_pattern = [
                    [4, 4, 4, 4, 4],
                    [4, 3, 2, 3, 4],
                    [4, 2, 0, 2, 4],
                    [4, 3, 2, 3, 4],
                    [4, 4, 4, 4, 4]
                ]
            
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if i < len(custom_pattern) and j < len(custom_pattern[i]):
                        self.grid_states[i,j] = custom_pattern[i][j]
        else:
            # Dynamic layout based on curriculum
            risk_probability = [0.15, 0.25, 0.4, 0.6][curriculum_level]
            center = self.grid_size[0]//2
            
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if i == center and j == center:
                        self.grid_states[i,j] = 0  # Start optimal
                    elif random.random() < risk_probability:
                        self.grid_states[i,j] = 4  # Risk zone
                    else:
                        if curriculum_level >= 2:
                            self.grid_states[i,j] = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
                        else:
                            self.grid_states[i,j] = random.randint(0, 3)

    def reset(self, seed=None, options=None):
        """Reset with better initialization"""
        super().reset(seed=seed)
        
        self.day = 0
        self.pest_level = self.config['initial_pest']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]
        self.last_action = None
        self.time_in_risk_zone = 0
        self.visited_positions = {tuple(self.current_pos)}
        self.step_count = 0
        self.cumulative_reward = 0
        self.best_pest_level = 1.0
        self.exploration_bonus = 0
        
        try:
            temp, humidity = get_weather(self.config['location'])
        except Exception:
            temp, humidity = 25.0, 65.0

        self.state = {
            "temp": np.array([temp], dtype=np.float32),
            "humidity": np.array([humidity], dtype=np.float32),
            "crop_type": random.randint(0, 2),
            "storage_method": random.randint(0, 2),
            "duration": np.array([0], dtype=np.float32),
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)],
            "nearby_zones": self._get_adjacent_zones(),
            "step_count": np.array([0], dtype=np.float32)
        }

        if self.render_mode == 'human' and self.visualizer:
            self.visualizer.reset()
            
        return self._get_observation(), {}

    def _get_observation(self):
        """Return observation based on selected observation type"""
        if self.obs_type == "mlp":
            return self._get_mlp_observation()
        elif self.obs_type == "cnn":
            return self._get_cnn_observation()
        else:  # multi
            return {
                "vector": self._get_mlp_observation(),
                "image": self._get_cnn_observation()
            }

    def _get_mlp_observation(self):
        """Simplified observation vector"""
        temp_norm = (self.state["temp"][0] - 20) / 20
        humidity_norm = (self.state["humidity"][0] - 60) / 30
        duration_norm = self.state["duration"][0] / self.config['max_days']
        pest_norm = self.state["pest_level"][0]
        pos_norm = np.array(self.current_pos, dtype=np.float32) / max(self.grid_size)
        zone_norm = self.state["zone_type"] / 4.0
        
        nearby_counts = np.bincount(self.state["nearby_zones"] + 1, minlength=6)[:5] / 4.0
        
        return np.concatenate([
            [temp_norm],
            [humidity_norm], 
            [self.state["crop_type"] / 2.0],
            [self.state["storage_method"] / 2.0],
            [duration_norm],
            [pest_norm],
            pos_norm,
            [zone_norm],
            nearby_counts[:3]
        ], dtype=np.float32)

    def _get_cnn_observation(self):
        """Enhanced CNN observation"""
        grid_img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                zone_type = self.grid_states[i,j]
                grid_img[i,j] = self.STATE_TYPES[zone_type]["color"]
        
        i, j = self.current_pos
        grid_img[i,j] = [255, 255, 255]
        
        img = cv2.resize(grid_img, (64, 64), interpolation=cv2.INTER_NEAREST)
        
        pest_width = int(self.pest_level * 64)
        img[0, :pest_width, 0] = 255
        
        temp_height = int(np.clip((self.state["temp"][0] - 10) / 30, 0, 1) * 64)
        img[:temp_height, 0, 2] = 255
        
        return img

    def step(self, action):
        """Improved step function with updated action handling"""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        reward = 0
        terminated = False
        truncated = False
        self.last_action = action
        self.step_count += 1
        
        prev_pest_level = self.pest_level
        prev_zone = self.state["zone_type"]
        
        # Handle actions with updated numbering
        if action < 17:  # Zone/treatment actions
            zone_actions = self.STATE_TYPES[self.state["zone_type"]]["actions"]
            if action in zone_actions:
                reward += self._handle_zone_action(action)
            else:
                reward -= 1.0  # Invalid action penalty
        else:  # Movement actions (17-20)
            new_pos = self._calculate_new_position(action)
            if new_pos != self.current_pos:
                reward += self._calculate_movement_reward(new_pos)
                self.current_pos = new_pos
                self.visited_positions.add(tuple(self.current_pos))
                if len(self.visited_positions) > self.exploration_bonus:
                    reward += 0.1
                    self.exploration_bonus = len(self.visited_positions)
            else:
                reward -= 0.1

        self._simulate_weather_change()
        self._update_pest_level()
        
        self.day += 1
        self.state.update({
            "duration": np.array([self.day], dtype=np.float32),
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)],
            "nearby_zones": self._get_adjacent_zones(),
            "step_count": np.array([self.step_count], dtype=np.float32)
        })

        # Enhanced reward shaping
        pest_improvement = prev_pest_level - self.pest_level
        if pest_improvement > 0:
            reward += pest_improvement * 10
        
        if self.pest_level < self.best_pest_level:
            reward += (self.best_pest_level - self.pest_level) * 5
            self.best_pest_level = self.pest_level
        
        suitability = self.calculate_zone_suitability()[tuple(self.current_pos)]
        reward += suitability * 0.5
        
        if self.pest_level < 0.8:
            reward += 0.2
        
        reward -= 0.05
        
        # Check termination conditions
        if self.pest_level >= 0.95:
            reward -= 10
            terminated = True
        elif self.day >= self.config['max_days']:
            reward += (1.0 - self.pest_level) * 5
            terminated = True
        elif self.last_action == 15:  # Emergency harvest
            reward += (1.0 - self.pest_level) * 8
            terminated = True
        elif self.state["zone_type"] == 4 and self.time_in_risk_zone > 3:
            reward -= 5
            terminated = True
        
        self.cumulative_reward += reward
        
        info = {
            "current_action": self.ACTIONS[action],
            "day": self.day,
            "position": tuple(self.current_pos),
            "zone_type": self.STATE_TYPES[self.state["zone_type"]]["name"],
            "suitability": float(suitability),
            "pest_level": float(self.pest_level),
            "time_in_risk": float(self.time_in_risk_zone),
            "visited": len(self.visited_positions),
            "cumulative_reward": float(self.cumulative_reward),
            "pest_improvement": float(pest_improvement)
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_adjacent_zones(self):
        """Get neighboring zone types"""
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        zones = []
        for di, dj in directions:
            ni, nj = self.current_pos[0]+di, self.current_pos[1]+dj
            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                zones.append(self.grid_states[ni,nj])
            else:
                zones.append(-1)
        return np.array(zones, dtype=np.int32)

    def _calculate_new_position(self, action):
        """Calculate new position after movement - updated for new action indices"""
        new_pos = self.current_pos.copy()
        if action == 17:  # Up
            new_pos[0] = max(0, new_pos[0]-1)
        elif action == 18:  # Down
            new_pos[0] = min(self.grid_size[0]-1, new_pos[0]+1)
        elif action == 19:  # Left
            new_pos[1] = max(0, new_pos[1]-1)
        elif action == 20:  # Right
            new_pos[1] = min(self.grid_size[1]-1, new_pos[1]+1)
        return new_pos

    def _calculate_movement_reward(self, new_pos):
        """Enhanced movement reward with strong risk zone incentives"""
        target_zone = self.grid_states[tuple(new_pos)]
        current_suitability = self.calculate_zone_suitability()[tuple(self.current_pos)]
        target_suitability = self.calculate_zone_suitability()[tuple(new_pos)]
        
        suitability_gain = target_suitability - current_suitability
        
        if target_zone == 4:  # Moving TO risk Zone
            self.time_in_risk_zone += 1
            base_reward = 0.5
            risk_penalty = -0.2 * self.time_in_risk_zone
            return base_reward + risk_penalty
        else:
            if self.state["zone_type"] == 4:  # Moving OUT of risk zone
                self.time_in_risk_zone = 0
                return 0.3 + (2.0 * suitability_gain)
            else:
                self.time_in_risk_zone = max(0, self.time_in_risk_zone - 1)
                return 0.1 + (1.5 * suitability_gain)

    def _handle_zone_action(self, action):
        """Handle zone-specific actions with improved effects"""
        zone_type = self.state["zone_type"]
        zone_info = self.STATE_TYPES[zone_type]
        
        if action not in zone_info["actions"]:
            return -1.0
        
        pest_change, action_reward = zone_info["action_effects"][action]
        
        # Apply action effects with some environmental variation
        actual_pest_change = pest_change * (0.8 + 0.4 * random.random())
        self.pest_level = np.clip(self.pest_level + actual_pest_change, 0, 1)
        
        return action_reward

    def calculate_zone_suitability(self):
        """Enhanced suitability calculation"""
        scores = np.zeros(self.grid_size)
        temp = self.state['temp'][0]
        humidity = self.state['humidity'][0]
        crop_type = self.state['crop_type']
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                zone = self.STATE_TYPES[self.grid_states[i,j]]
                
                if zone['name'] == "Risk Zone":
                    scores[i,j] = 0
                    continue
                
                if zone['ideal_temp'] is None:  # Risk zone
                    scores[i,j] = 0
                    continue
                
                temp_min, temp_max = zone['ideal_temp']
                if temp_min <= temp <= temp_max:
                    temp_score = 1.0
                else:
                    temp_score = max(0, 1 - abs(temp - (temp_min + temp_max)/2) / 10)
                
                hum_min, hum_max = zone['ideal_humidity']
                if hum_min <= humidity <= hum_max:
                    humidity_score = 1.0
                else:
                    humidity_score = max(0, 1 - abs(humidity - (hum_min + hum_max)/2) / 20)
                
                crop_multiplier = [1.0, 0.9, 1.1][crop_type]
                
                scores[i,j] = (temp_score * 0.5 + humidity_score * 0.5) * crop_multiplier
                
        return scores

    def _simulate_weather_change(self):
        """More stable weather changes"""
        temp_change = random.uniform(-1, 1)
        humidity_change = random.uniform(-3, 3)
        
        self.state["temp"] = np.clip(
            self.state["temp"] + temp_change,
            15, 35
        )
        self.state["humidity"] = np.clip(
            self.state["humidity"] + humidity_change,
            40, 85
        )

    def _update_pest_level(self):
        """More balanced pest growth"""
        base_risk = 0.02
        temp = self.state["temp"][0]
        humidity = self.state["humidity"][0]
        
        temp_effect = max(0, (temp - 25) * 0.005)
        humidity_effect = max(0, (humidity - 65) * 0.008)
        
        storage_risk = [0.0, 0.01, 0.02][self.state["storage_method"]]
        crop_factor = [0.8, 1.0, 1.1][self.state["crop_type"]]
        zone_risk = {0: 0.3, 1: 0.5, 2: 0.6, 3: 0.7, 4: 1.5}[self.state["zone_type"]]
        
        daily_increase = (base_risk + temp_effect + humidity_effect + storage_risk) * crop_factor * zone_risk
        self.pest_level = np.clip(self.pest_level + daily_increase, 0, 1)

    def _init_visualization(self):
        """Initialize visualization"""
        try:
            self.visualizer = StorageVisualizer(self)
        except ImportError as e:
            print(f"Visualization disabled: {str(e)}")
            self.visualizer = None
            self.render_mode = 'console'

    def render(self):
        """Render environment"""
        if self.render_mode == 'human' and self.visualizer:
            return self.visualizer.render()
        elif self.render_mode == 'console' or self.render_mode is None:
            self._console_render()
            return None
        elif self.render_mode == 'rgb_array':
            if self.visualizer:
                return self.visualizer.get_rgb_array()
            return self._get_cnn_observation()

    def _console_render(self):
        """Enhanced console rendering"""
        print(f"\n=== Day {self.day}/{self.config['max_days']} ===")
        print(f"Position: {self.current_pos} | Zone: {self.STATE_TYPES[self.state['zone_type']]['name']}")
        print(f"Zone Description: {self.STATE_TYPES[self.state['zone_type']]['description']}")
        print(f"Temp: {self.state['temp'][0]:.1f}°C | Humidity: {self.state['humidity'][0]:.1f}%")
        print(f"Crop: {self.CROP_TYPES[self.state['crop_type']]} | Storage: {self.STORAGE_METHODS[self.state['storage_method']]}")
        
        pest_bar = '█' * int(self.pest_level * 20) + '░' * (20 - int(self.pest_level * 20))
        print(f"Pest Risk: {pest_bar} {self.pest_level*100:.1f}%")
        print(f"Suitability: {self.calculate_zone_suitability()[tuple(self.current_pos)]:.2f}")
        print(f"Cumulative Reward: {self.cumulative_reward:.2f}")
        
        # Show recommended actions for current zone
        zone_type = self.state["zone_type"]
        zone_info = self.STATE_TYPES[zone_type]
        # Filter out movement actions (17-20) but keep all treatment actions (0-16)
        treatment_actions = [a for a in zone_info["actions"] if a <= 16]
        recommended_actions = [self.ACTIONS[a] for a in treatment_actions]
        
        print(f"Available Actions ({len(treatment_actions)} total):")
        for i, action_idx in enumerate(treatment_actions):
            print(f"  {i+1}. {self.ACTIONS[action_idx]} (Action {action_idx})")
        
        if not recommended_actions:
            print("⚠️  WARNING: No valid treatment actions available for this zone!")
        
        # Show why these actions are recommended
        if zone_type == 0:
            print("→ Focus: Maintain perfect conditions with minimal intervention")
        elif zone_type == 1:
            print("→ Focus: Add moisture to prevent grain brittleness and dust")
        elif zone_type == 2:
            print("→ Focus: Leverage good airflow to prevent mold and fungus")
        elif zone_type == 3:
            print("→ Focus: Remove excess moisture before mold develops")
        elif zone_type == 4:
            print("→ Focus: EMERGENCY - Save crop from active pest damage")
            print("→ These are CRITICAL INTERVENTIONS - choose quickly!")
        
        if hasattr(self, 'last_action') and self.last_action is not None:
            print(f"Last Action: {self.ACTIONS[self.last_action]}")
            
        # Debug info for all zones to see what's happening
        print(f"DEBUG - Zone {zone_type} raw actions: {zone_info['actions']}")
        print(f"DEBUG - Filtered treatment actions: {treatment_actions}")
        print(f"DEBUG - Action effects keys: {list(zone_info['action_effects'].keys())}")
        print(f"DEBUG - Total ACTIONS array length: {len(self.ACTIONS)}")
        
        # Let's also check if action indices are valid
        for action_idx in zone_info['actions']:
            if action_idx < len(self.ACTIONS):
                print(f"DEBUG - Action {action_idx}: {self.ACTIONS[action_idx]}")
            else:
                print(f"DEBUG - INVALID Action {action_idx}: OUT OF RANGE!")

    def close(self):
        """Cleanup"""
        if self.visualizer:
            self.visualizer.close()