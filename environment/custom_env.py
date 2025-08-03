import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Tuple, List
from environment.rendering import StorageVisualizer
import cv2

class StorageEnv(gym.Env):
    """A Farm Storage Environment consisted of Condition Analysis & Navigation Learning, and Treatment Application"""

    # Constants
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    
    STATE_TYPES = [
        {   # 0 - Optimal Storage
            "name": "Optimal Storage", 
            "color": (100, 255, 100),
            "ideal_temp": (18, 25), 
            "ideal_humidity": (60, 70),
            "ideal_pest": (0.0, 0.2),
            "description": "Perfect balanced conditions - maintain with minimal intervention",
            "actions": [0, 6, 7],  # Monitor, temperature control, preventive neem
        },
        {   # 1 - Too Dry (Low humidity, may cause grain brittleness)
            "name": "Too Dry", 
            "color": (100, 200, 255),
            "ideal_temp": (15, 30), 
            "ideal_humidity": (30, 50),
            "ideal_pest": (0.0, 0.3),
            "description": "Dangerously low humidity - grain may crack, dust increases pest attraction",
            "actions": [2, 4, 8, 12],  # Reduce ventilation, add controlled moisture, moisture control, sealed containers
        },
        {   # 2 - Well Ventilated
            "name": "Well Ventilated", 
            "color": (200, 255, 200),
            "ideal_temp": (20, 32), 
            "ideal_humidity": (50, 70),
            "ideal_pest": (0.0, 0.4),
            "description": "Excellent airflow - perfect for preventing mold and fungal issues",
            "actions": [0, 1, 5, 10],  # Monitor, increase ventilation, solar drying, aromatic herbs
        },
        {   # 3 - High Humidity
            "name": "High Humidity", 
            "color": (255, 255, 100),
            "ideal_temp": (15, 35), 
            "ideal_humidity": (70, 90),
            "ideal_pest": (0.0, 0.6),
            "description": "Dangerous moisture levels - mold and fungal growth imminent",
            "actions": [1, 3, 5, 9, 11],  # Increase ventilation, moisture absorbers, solar drying, ash, DE
        },
        {   # 4 - Critical Risk Zone
            "name": "Critical Risk", 
            "color": (255, 50, 50),
            "ideal_temp": (0, 50), 
            "ideal_humidity": (0, 100),
            "ideal_pest": (0.5, 1.0),
            "description": "EMERGENCY: Active pest infestation or severe environmental damage detected",
            "actions": [11, 13, 14, 15, 16],  # Emergency protocols only
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
        "Move right",                    # 20
        "Read conditions"                # 21 - Analyze current conditions
    ]

    # GROUND TRUTH MAPPING: Condition ranges to correct zone
    CONDITION_ZONE_MAPPING = [
        # Format: (temp_min, temp_max, humidity_min, humidity_max, pest_min, pest_max) -> zone_index
        
        # OPTIMAL STORAGE CONDITIONS
        (18, 25, 60, 70, 0.0, 0.2, 0),  # Perfect conditions → Optimal Storage
        (20, 24, 62, 68, 0.0, 0.15, 0), # Even more perfect → Optimal Storage
        
        # TOO DRY CONDITIONS
        (15, 30, 30, 50, 0.0, 0.3, 1),  # Low humidity → Too Dry
        (25, 35, 35, 45, 0.1, 0.25, 1), # Hot and dry → Too Dry
        
        # WELL VENTILATED CONDITIONS
        (20, 32, 50, 70, 0.0, 0.4, 2),  # Good airflow conditions → Well Ventilated
        (22, 28, 55, 65, 0.1, 0.3, 2),  # Moderate conditions → Well Ventilated
        
        # HIGH HUMIDITY CONDITIONS
        (15, 35, 70, 90, 0.0, 0.6, 3),  # High moisture → High Humidity
        (28, 35, 75, 85, 0.2, 0.5, 3),  # Hot and humid → High Humidity
        
        # CRITICAL RISK CONDITIONS
        (0, 50, 0, 100, 0.5, 1.0, 4),   # High pest level → Critical Risk
        (35, 50, 80, 100, 0.3, 1.0, 4), # Extreme conditions → Critical Risk
    ]
    
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None, obs_type: str = "multi"):
        super().__init__()
        
        self.config = config or {
            'max_steps': 20,  # 20 for better exploration
            'initial_pest': 0.1,
            'location': "Kigali",
            'grid_size': (5, 5),
            'layout': 'custom',
            'obs_type': 'multi',
            'curriculum_stage': 1,  # 1=easy, 2=medium, 3=full
            'use_action_masking': True,  # Enabled action masking
            'intermediate_rewards': True  # Enabled intermediate navigation rewards
        }
        
        self.obs_type = self.config.get('obs_type', obs_type)
        self.render_mode = render_mode
        self.visualizer = None
        self.grid_size = self.config['grid_size']
        self.current_pos = [self.grid_size[0]//2, self.grid_size[1]//2]
        self.step_count = 0
        
        # Episode phase tracking with enhanced progress monitoring
        self.episode_phase = "ANALYZE"  # ANALYZE -> NAVIGATE -> TREAT -> END
        self.has_read_conditions = False
        self.target_zone = None
        self.chosen_zone = None
        self.conditions_analyzed = False
        
        # Episode progress tracking for intermediate rewards
        self.episode_progress = {
            'conditions_read': False,
            'started_navigation': False,
            'reached_any_zone': False,
            'reached_target_zone': False,
            'applied_treatment': False
        }
        
        # Distance tracking for navigation rewards
        self.previous_distance_to_target = None
        self.best_distance_to_target = float('inf')
        
        self._init_grid_states()
        if self.render_mode == 'human':
            self._init_visualization()

        # Enhanced observation space with phase encoding
        if self.obs_type == "mlp":
            self.observation_space = spaces.Box(
                low=-10, high=100, 
                shape=(13,),
                dtype=np.float32
            )
        elif self.obs_type == "cnn":
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3),
                dtype=np.uint8
            )
        else:  # multi-Input Policy
            self.observation_space = spaces.Dict({
                "vector": spaces.Box(low=-10, high=100, shape=(13,), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "action_mask": spaces.Box(low=0, high=1, shape=(len(self.ACTIONS),), dtype=np.int8)
            })

        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def _init_grid_states(self):
        """Create a grid with all 5 zone types"""
        self.grid_states = np.full(self.grid_size, -1, dtype=int)
        
        rows, cols = self.grid_size
        
        # Placing all 5 zones in corners and center
        zone_positions = [
            (0, 0),                 # Top-left - Optimal Storage
            (0, cols-1),            # Top-right - Too Dry  
            (rows-1, 0),            # Bottom-left - Ventilated
            (rows-1, cols-1),       # Bottom-right - High Humidity
            (rows//2, cols//2)      # Critical Risk in center
        ]
        
        for idx, pos in enumerate(zone_positions):
            if idx < len(self.STATE_TYPES):
                self.grid_states[pos] = idx

    def _get_empty_positions(self):
        """Get all empty grid positions (where grid_states == -1)"""
        empty_positions = []
        rows, cols = self.grid_size
        
        for row in range(rows):
            for col in range(cols):
                if self.grid_states[row, col] == -1:  # Empty cell
                    empty_positions.append([row, col])
        
        return empty_positions

    def _get_adjacent_empty_positions(self, target_pos: List[int]) -> List[List[int]]:
        """Get empty positions adjacent to target position"""
        adjacent_positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            new_row, new_col = target_pos[0] + dr, target_pos[1] + dc
            if (0 <= new_row < self.grid_size[0] and 
                0 <= new_col < self.grid_size[1] and 
                self.grid_states[new_row, new_col] == -1):
                adjacent_positions.append([new_row, new_col])
        return adjacent_positions

    #Defining agent starting position
    def _get_random_empty_position(self):
        """Get a random empty position from available empty cells"""
        empty_positions = self._get_empty_positions()
        
        if not empty_positions:
            # Fallback: if no empty positions, use center
            print("Warning: No empty positions available, using center")
            return [self.grid_size[0]//2, self.grid_size[1]//2]
        
        return random.choice(empty_positions)

    def _get_curriculum_start_position(self) -> List[int]:
        """Get starting position based on curriculum stage"""
        curriculum_stage = self.config.get('curriculum_stage', 1)
        
        if curriculum_stage == 1:  # Easy: Start adjacent to target zone
            target_positions = np.where(self.grid_states == self.target_zone)
            if len(target_positions[0]) > 0:
                target_pos = [target_positions[0][0], target_positions[1][0]]
                adjacent_positions = self._get_adjacent_empty_positions(target_pos)
                if adjacent_positions:
                    return random.choice(adjacent_positions)
        
        elif curriculum_stage == 2:  # Medium: Start within 2 steps of target
            target_positions = np.where(self.grid_states == self.target_zone)
            if len(target_positions[0]) > 0:
                target_pos = [target_positions[0][0], target_positions[1][0]]
                nearby_positions = []
                for empty_pos in self._get_empty_positions():
                    distance = abs(empty_pos[0] - target_pos[0]) + abs(empty_pos[1] - target_pos[1])
                    if distance <= 2:
                        nearby_positions.append(empty_pos)
                if nearby_positions:
                    return random.choice(nearby_positions)
        
        # Stage 3 (full) or fallback: Random empty position
        return self._get_random_empty_position()

    def _determine_correct_zone(self, temp: float, humidity: float, pest: float) -> int:
        """Determine the correct zone based on current conditions using ground truth mapping"""
        for temp_min, temp_max, hum_min, hum_max, pest_min, pest_max, zone_idx in self.CONDITION_ZONE_MAPPING:
            if (temp_min <= temp <= temp_max and 
                hum_min <= humidity <= hum_max and 
                pest_min <= pest <= pest_max):
                return zone_idx
        
        # Fallback logic if no exact match
        if pest >= 0.5:
            return 4  # Critical Risk
        elif humidity >= 70:
            return 3  # High Humidity
        elif humidity <= 50:
            return 1  # Too Dry
        elif 20 <= temp <= 32:
            return 2  # Well Ventilated
        else:
            return 0  # Optimal Storage

    def _calculate_distance_to_target(self) -> float:
        """Calculate Manhattan distance to target zone"""
        if self.target_zone is None:
            return 0.0
        
        target_positions = np.where(self.grid_states == self.target_zone)
        if len(target_positions[0]) == 0:
            return 0.0
        
        min_distance = float('inf')
        for target_row, target_col in zip(target_positions[0], target_positions[1]):
            distance = abs(self.current_pos[0] - target_row) + abs(self.current_pos[1] - target_col)
            min_distance = min(min_distance, distance)
        
        return min_distance

    def get_valid_actions(self) -> List[int]:
        """Get valid actions based on current phase (for action masking)"""
        if not self.config.get('use_action_masking', False):
            return list(range(len(self.ACTIONS)))
        
        if self.episode_phase == "ANALYZE":
            return [21]  # Only read conditions
        
        elif self.episode_phase == "NAVIGATE":
            return [17, 18, 19, 20]  # Only movement actions
        
        elif self.episode_phase == "TREAT":
            current_zone = self.grid_states[tuple(self.current_pos)]
            if current_zone != -1:
                return self.STATE_TYPES[current_zone]["actions"]
            return []
        
        return list(range(len(self.ACTIONS)))

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for current state"""
        mask = np.zeros(len(self.ACTIONS), dtype=np.int8)
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = 1
        return mask

    def reset(self, seed=None, options=None):
        """Reset with enhanced curriculum learning and progress tracking"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_phase = "ANALYZE"
        self.has_read_conditions = False
        self.conditions_analyzed = False
        self.chosen_zone = None
        
        # Reset progress tracking
        self.episode_progress = {
            'conditions_read': False,
            'started_navigation': False,
            'reached_any_zone': False,
            'reached_target_zone': False,
            'applied_treatment': False
        }
        
        # Generate random conditions
        self.temp = random.uniform(15, 40)
        self.humidity = random.uniform(30, 90)
        self.pest_level = random.uniform(0.0, 0.8)
        
        # Determine the correct zone for these conditions
        self.target_zone = self._determine_correct_zone(self.temp, self.humidity, self.pest_level)
        
        # Use curriculum-based starting position
        self.current_pos = self._get_curriculum_start_position()
        
        # Initialize distance tracking
        self.previous_distance_to_target = self._calculate_distance_to_target()
        self.best_distance_to_target = self.previous_distance_to_target
        
        # Initialize state
        self.state = {
            "temp": np.array([self.temp], dtype=np.float32),
            "humidity": np.array([self.humidity], dtype=np.float32),
            "crop_type": random.randint(0, 2),
            "storage_method": random.randint(0, 2),
            "pest_level": np.array([self.pest_level], dtype=np.float32),
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)],
            "phase": self.episode_phase,
            "target_zone": self.target_zone
        }

        if self.render_mode == 'human' and self.visualizer:
            self.visualizer.reset()
            
        return self._get_observation(), {}

    def step(self, action):
        """A step function with intermediate rewards and learning signals"""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        reward = 0
        terminated = False
        truncated = False
        self.step_count += 1

        info = {
            "phase": self.episode_phase,
            "target_zone": self.target_zone,
            "chosen_zone": self.chosen_zone,
            "correct_choice": False,
            "conditions": {
                "temp": float(self.temp),
                "humidity": float(self.humidity), 
                "pest": float(self.pest_level)
            },
            "step_count": self.step_count,
            "valid_actions": self.get_valid_actions(),
            "episode_progress": self.episode_progress.copy()
        }

        # Check if action is valid (for action masking)
        valid_actions = self.get_valid_actions()
        if self.config.get('use_action_masking', False) and action not in valid_actions:
            reward -= 1.0  # Penalty for invalid action
            info["message"] = f"Invalid action {action} for phase {self.episode_phase}"
            # Don't terminate, just penalize
        else:
            # PHASE 1: ANALYZE CONDITIONS
            if self.episode_phase == "ANALYZE":
                if action == 21:  # Read conditions
                    self.has_read_conditions = True
                    self.conditions_analyzed = True
                    self.episode_phase = "NAVIGATE"
                    
                    # Progress tracking and rewards
                    if not self.episode_progress['conditions_read']:
                        self.episode_progress['conditions_read'] = True
                        reward += 2.0  # Good reward for reading conditions
                        info["message"] = f"Conditions analyzed: T={self.temp:.1f}°C, H={self.humidity:.1f}%, P={self.pest_level:.2f}"
                    
                else:
                    reward -= 0.5  # Penalty for not reading conditions first
                    info["message"] = "Must read conditions first (action 21)"

            # PHASE 2: NAVIGATE TO ZONE
            elif self.episode_phase == "NAVIGATE":
                if action in [17, 18, 19, 20]:  # Movement actions
                    # Track that navigation has started
                    if not self.episode_progress['started_navigation']:
                        self.episode_progress['started_navigation'] = True
                        reward += 0.5  # Small reward for starting navigation
                    
                    new_pos = self._calculate_new_position(action)
                    if (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                        self.current_pos = new_pos
                        
                        # INTERMEDIATE NAVIGATION REWARDS
                        if self.config.get('intermediate_rewards', True):
                            current_distance = self._calculate_distance_to_target()
                            
                            # Reward for getting closer to target
                            if current_distance < self.previous_distance_to_target:
                                reward += 0.3  # Moving closer reward
                                info["message"] = f"Moving closer to target (distance: {current_distance})"
                            elif current_distance > self.previous_distance_to_target:
                                reward -= 0.1  # Moving away penalty (small)
                                info["message"] = f"Moving away from target (distance: {current_distance})"
                            else:
                                reward += 0.05  # Small reward for valid movement
                                info["message"] = "Valid movement"
                            
                            # Track best distance reached
                            if current_distance < self.best_distance_to_target:
                                self.best_distance_to_target = current_distance
                                reward += 0.2  # Bonus for reaching new best distance
                            
                            self.previous_distance_to_target = current_distance
                        
                        # Check if agent reached a zone
                        current_zone = self.grid_states[tuple(self.current_pos)]
                        if current_zone != -1:  # Reached a zone
                            self.chosen_zone = current_zone
                            self.episode_phase = "TREAT"
                            
                            # Progress tracking
                            if not self.episode_progress['reached_any_zone']:
                                self.episode_progress['reached_any_zone'] = True
                                reward += 1.0  # Reward for reaching any zone
                            
                            # BONUS for reaching the CORRECT zone
                            if current_zone == self.target_zone:
                                if not self.episode_progress['reached_target_zone']:
                                    self.episode_progress['reached_target_zone'] = True
                                    reward += 3.0  # BIG bonus for reaching target zone
                                    info["message"] = f"SUCCESS! Reached correct zone: {self.STATE_TYPES[current_zone]['name']}"
                                else:
                                    reward += 1.0
                            else:
                                reward += 0.5  # Small reward for reaching wrong zone
                                info["message"] = f"Reached wrong zone: {self.STATE_TYPES[current_zone]['name']} (target: {self.STATE_TYPES[self.target_zone]['name']})"
                        
                    else:
                        reward -= 0.3  # Penalty for hitting boundary
                        info["message"] = "Cannot move outside grid"
                else:
                    reward -= 0.5  # Penalty for wrong action type
                    info["message"] = "Must navigate to a zone (actions 17-20)"

            # PHASE 3: APPLY TREATMENT
            elif self.episode_phase == "TREAT":
                current_zone = self.grid_states[tuple(self.current_pos)]
                zone_info = self.STATE_TYPES[current_zone]
                
                if action in zone_info["actions"]:
                    # Progress tracking
                    if not self.episode_progress['applied_treatment']:
                        self.episode_progress['applied_treatment'] = True
                        reward += 1.0  # Reward for applying any valid treatment
                    
                    # CHECK IF AGENT CHOSE CORRECT ZONE
                    if self.chosen_zone == self.target_zone:
                        reward += 8.0  # BIG REWARD for correct zone choice
                        info["correct_choice"] = True
                        info["message"] = f"SUCCESS! Correct zone and treatment applied"
                    else:
                        reward -= 4.0  # PENALTY for wrong zone choice
                        info["message"] = f"WRONG ZONE! Should be {self.STATE_TYPES[self.target_zone]['name']}"
                    
                    terminated = True  # Episode ends after treatment
                    
                else:
                    reward -= 2.0  # Penalty for invalid treatment
                    info["message"] = f"Invalid treatment for {zone_info['name']} zone"
                    terminated = True  # End episode on invalid treatment

        # Update state
        self.state.update({
            "position": np.array(self.current_pos, dtype=np.int32),
            "zone_type": self.grid_states[tuple(self.current_pos)],
            "phase": self.episode_phase
        })

        # Truncation check
        if self.step_count >= self.config['max_steps']:
            truncated = True
            reward -= 2.0  # penalty for taking too long
            info["message"] = "Episode truncated - took too many steps"

        # Add small time penalty to encourage efficiency
        reward -= 0.02  # Very small step penalty

        info["reward"] = float(reward)
        info["target_zone_name"] = self.STATE_TYPES[self.target_zone]["name"]
        if self.chosen_zone is not None:
            info["chosen_zone_name"] = self.STATE_TYPES[self.chosen_zone]["name"]

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """Return observation based on selected observation type"""
        if self.obs_type == "mlp":
            return self._get_mlp_observation()
        elif self.obs_type == "cnn":
            return self._get_cnn_observation()
        else:  # multi
            return {
                "vector": self._get_mlp_observation(),
                "image": self._get_cnn_observation(),
                "action_mask": self._get_action_mask()
            }

    def _get_mlp_observation(self):
        """Enhanced observation vector with better feature encoding"""
        temp_norm = (self.temp - 25) / 15  # Normalize around 25°C
        humidity_norm = (self.humidity - 60) / 30  # Normalize around 60%
        pest_norm = self.pest_level  # Already 0-1
        pos_norm = np.array(self.current_pos, dtype=np.float32) / max(self.grid_size)
        
        # One-hot encoding for phases
        phase_encoding = np.zeros(3, dtype=np.float32)
        phase_idx = {"ANALYZE": 0, "NAVIGATE": 1, "TREAT": 2}[self.episode_phase]
        phase_encoding[phase_idx] = 1.0
        
        zone_norm = self.state["zone_type"] / 4.0 if self.state["zone_type"] != -1 else -0.25
        
        # Distance to target zone (if conditions have been read)
        distance_to_target = 0.0
        if self.has_read_conditions:
            distance_to_target = self._calculate_distance_to_target() / max(self.grid_size)
        
        return np.array([
            temp_norm,
            humidity_norm,
            pest_norm,
            self.state["crop_type"] / 2.0,
            self.state["storage_method"] / 2.0,
            pos_norm[0],
            pos_norm[1], 
            zone_norm,
            float(self.has_read_conditions),
            distance_to_target,
            *phase_encoding  # Expand the 3-element phase encoding
        ], dtype=np.float32)

    def _get_cnn_observation(self):
        """Enhanced CNN observation with better visual cues"""
        grid_img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        
        # Draw zones
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                zone_type = self.grid_states[i,j]
                if zone_type != -1:
                    grid_img[i,j] = self.STATE_TYPES[zone_type]["color"]
                else:
                    grid_img[i,j] = (50, 50, 50)  # Empty cells
        
        # Draw agent position with phase-based color
        i, j = self.current_pos
        phase_colors = {
            "ANALYZE": [255, 255, 255],    # White
            "NAVIGATE": [255, 255, 0],     # Yellow
            "TREAT": [255, 0, 255]         # Magenta
        }
        grid_img[i,j] = phase_colors.get(self.episode_phase, [255, 255, 255])
        
        # Highlight target zone if conditions have been read
        if self.has_read_conditions:
            target_positions = np.where(self.grid_states == self.target_zone)
            for ti, tj in zip(target_positions[0], target_positions[1]):
                # Add bright border to target zone
                original_color = grid_img[ti,tj]
                grid_img[ti,tj] = [min(255, c + 50) for c in original_color]  # Brighten target
        
        # Resize to standard size
        img = cv2.resize(grid_img, (64, 64), interpolation=cv2.INTER_NEAREST)
        
        return img

    def _calculate_new_position(self, action):
        """Calculate new position after movement"""
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
        """Enhanced console rendering with progress tracking and curriculum info"""
        print(f"\n=== EPISODE STEP {self.step_count}/{self.config['max_steps']} ===")
        print(f"Phase: {self.episode_phase} | Curriculum Stage: {self.config.get('curriculum_stage', 1)}")
        print(f"Position: {self.current_pos}")
        
        current_zone = self.grid_states[tuple(self.current_pos)]
        if current_zone != -1:
            print(f"Current Zone: {self.STATE_TYPES[current_zone]['name']}")
        else:
            print("Current Zone: Empty space")
        
        # Show conditions if read
        if self.has_read_conditions:
            print(f"\n CONDITIONS ANALYZED:")
            print(f" Temperature: {self.temp:.1f}°C")
            print(f" Humidity: {self.humidity:.1f}%")
            print(f" Pest Level: {self.pest_level:.2f}")
            print(f" Target Zone: {self.STATE_TYPES[self.target_zone]['name']}")
            
            # Show distance to target during navigation
            if self.episode_phase == "NAVIGATE":
                distance = self._calculate_distance_to_target()
                print(f"  Distance to Target: {distance} steps")
                print(f"  Best Distance: {self.best_distance_to_target} steps")
        else:
            print(f"\n Conditions not yet analyzed")
        
        # Show episode progress
        print(f"\n EPISODE PROGRESS:")
        progress_items = [
            ("Conditions Read", self.episode_progress['conditions_read']),
            ("Started Navigation", self.episode_progress['started_navigation']),
            ("Reached Any Zone", self.episode_progress['reached_any_zone']),
            ("Reached Target Zone", self.episode_progress['reached_target_zone']),
            ("Applied Treatment", self.episode_progress['applied_treatment'])
        ]
        
        for item, completed in progress_items:
            status = "Great" if completed else "Needs Improvement"
            print(f"   {status} {item}")
        
        # Show chosen zone if any
        if self.chosen_zone is not None:
            print(f"\n Chosen Zone: {self.STATE_TYPES[self.chosen_zone]['name']}")
            if self.chosen_zone == self.target_zone:
                print(f"  CORRECT CHOICE!")
            else:
                print(f"  WRONG! Should be {self.STATE_TYPES[self.target_zone]['name']}")
        
        # Show available actions based on phase (with action masking info)
        valid_actions = self.get_valid_actions()
        print(f"\n AVAILABLE ACTIONS (Action Masking: {'ON' if self.config.get('use_action_masking', False) else 'OFF'}):")
        
        if self.episode_phase == "ANALYZE":
            print("   21. Read conditions (REQUIRED FIRST)")
        elif self.episode_phase == "NAVIGATE":
            print("   17. Move up    | 18. Move down")
            print("   19. Move left  | 20. Move right")
            if self.config.get('intermediate_rewards', True):
                print(" Tip: Moving closer to target zone gives more reward!")
        elif self.episode_phase == "TREAT":
            if current_zone != -1:
                zone_actions = self.STATE_TYPES[current_zone]["actions"]
                print(f"   Available treatments in {self.STATE_TYPES[current_zone]['name']}:")
                for action_idx in zone_actions:
                    print(f"   {action_idx}. {self.ACTIONS[action_idx]}")
            else:
                print("   No valid actions in empty cell!")
        
        # Show curriculum stage info
        curriculum_stage = self.config.get('curriculum_stage', 1)
        stage_descriptions = {
            1: "Easy - Start adjacent to target zone",
            2: "Medium - Start within 2 steps of target",
            3: "Full - Random start position"
        }
        print(f"\n Curriculum Stage {curriculum_stage}: {stage_descriptions.get(curriculum_stage, 'Unknown')}")
        
        print()

    def get_success_metrics(self) -> Dict:
        """Get detailed success metrics for training monitoring"""
        return {
            'conditions_read': self.episode_progress['conditions_read'],
            'started_navigation': self.episode_progress['started_navigation'],
            'reached_any_zone': self.episode_progress['reached_any_zone'],
            'reached_target_zone': self.episode_progress['reached_target_zone'],
            'applied_treatment': self.episode_progress['applied_treatment'],
            'chose_correct_zone': self.chosen_zone == self.target_zone if self.chosen_zone is not None else False,
            'episode_length': self.step_count,
            'best_distance_achieved': self.best_distance_to_target,
            'final_distance_to_target': self._calculate_distance_to_target(),
            'curriculum_stage': self.config.get('curriculum_stage', 1),
            'phase_reached': self.episode_phase
        }

    def update_curriculum(self, success_rate: float, episodes_completed: int):
        """Automatically progress curriculum based on success rate"""
        current_stage = self.config.get('curriculum_stage', 1)
        
        # Progress criteria
        if current_stage == 1 and success_rate > 0.7 and episodes_completed > 1000:
            self.config['curriculum_stage'] = 2
            print(f" Curriculum advanced to Stage 2 (Medium difficulty)")
        elif current_stage == 2 and success_rate > 0.6 and episodes_completed > 5000:
            self.config['curriculum_stage'] = 3
            print(f" Curriculum advanced to Stage 3 (Full difficulty)")

    def close(self):
        """Clean up resources"""
        if self.visualizer:
            self.visualizer.close()
        if hasattr(self, 'pygame') and self.render_mode == 'human':
            import pygame
            pygame.quit()