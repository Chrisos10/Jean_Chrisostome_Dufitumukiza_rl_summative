import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Tuple

import pygame  # For visualization

from environment.my_weather import get_weather, WeatherSimulator


class StorageEnv(gym.Env):
    CROP_TYPES = ["Maize", "Beans", "Rice"]
    STORAGE_METHODS = ["Silo", "Bags", "Traditional"]
    ACTIONS = [
        "Move North", "Move Northeast", "Move East", "Move Southeast",
        "Move South", "Move Southwest", "Move West", "Move Northwest",
        "Apply Treatment"
    ]

    metadata = {"render.modes": ["human", "rgb_array", "console"]}

    def __init__(self, config: Optional[Dict] = None, render_mode: Optional[str] = None):
        super().__init__()
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

        self.grid_size = 5
        self.agent_pos = np.array([2, 2])
        self.state_info = self._create_state_map()

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 10, 25, 0, 0]),
            high=np.array([4, 4, 1, 40, 95, 2, 2]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(9)

        self.weather_sim = WeatherSimulator(location=self.config['location'])
        self.weather_sim.use_real_api = False

        self.render_mode = render_mode
        self.visualizer = None
        if self.render_mode == 'human':
            self._init_visualization()

    def _init_visualization(self):
        try:
            self.visualizer = StorageVisualizer(self)
        except ImportError as e:
            print(f"Visualization disabled: {str(e)}")
            self.visualizer = None
            self.render_mode = "console"

    def _create_state_map(self) -> Dict[Tuple[int, int], Dict]:
        state_map = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                base_risk = 0.1 + (abs(x - 2) + abs(y - 2)) * 0.15
                state_map[(x, y)] = {
                    "condition": self._get_condition_label(base_risk),
                    "intervention": self._get_intervention(base_risk),
                    "base_risk": np.array([base_risk], dtype=np.float32)
                }
        return state_map

    def _get_condition_label(self, risk: float) -> str:
        if risk > 0.7:
            return "Critical"
        elif risk > 0.4:
            return "Moderate"
        return "Safe"

    def _get_intervention(self, risk: float) -> str:
        if risk > 0.7:
            return "Emergency Harvest"
        elif risk > 0.5:
            return "Apply Treatment"
        elif risk > 0.3:
            return "Increase Ventilation"
        return "Monitor"

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.agent_pos = np.array([2, 2])
        self.day = 0

        try:
            self.temp, self.humidity = get_weather(location=self.config['location'], grid_pos=tuple(self.agent_pos))
            self.temp = float(self.temp)
            self.humidity = float(self.humidity)
        except Exception as e:
            print(f"Weather error: {e}. Using defaults.")
            self.temp = 28.0
            self.humidity = 65.0

        self.crop_type = random.randint(0, 2)
        self.storage_method = random.randint(0, 2)
        self._update_state()

        if self.visualizer:
            self.visualizer.reset()

        return self._flatten_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = 0
        terminated = False
        prev_pos = self.agent_pos.copy()

        if action < 8:
            moves = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                     (1, 0), (1, -1), (0, -1), (-1, -1)]
            dx, dy = moves[action]
            new_pos = np.clip(self.agent_pos + [dx, dy], 0, self.grid_size - 1)
            self.agent_pos = new_pos

            reward = -0.1
            prev_risk = self.state_info[tuple(prev_pos)]["base_risk"][0]
            new_risk = self.state_info[tuple(new_pos)]["base_risk"][0]
            if new_risk > prev_risk:
                reward -= 0.2 * (new_risk - prev_risk)
        else:
            current_risk = self._get_current_risk()
            reward = 2 * current_risk
            self.state_info[tuple(self.agent_pos)]["base_risk"][0] *= 0.7

        self.day += 1
        self.temp, self.humidity = get_weather(
            location=self.config['location'],
            grid_pos=tuple(self.agent_pos))
        self.temp = float(self.temp)
        self.humidity = float(self.humidity)

        self._update_state()
        terminated = self.day >= self.config['max_days']

        info = {
            "day": self.day,
            "position": tuple(self.agent_pos),
            "condition": self.state_info[tuple(self.agent_pos)]["condition"],
            "intervention": self.state_info[tuple(self.agent_pos)]["intervention"],
            "weather": (self.temp, self.humidity),
            "recommended": self._get_recommended_action()
        }

        return self._flatten_state(), reward, terminated, False, info

    def _update_state(self):
        current_risk = self._get_current_risk()
        self.current_state = {
            "position": self.agent_pos.copy(),
            "risk_level": np.array([current_risk], dtype=np.float32),
            "temp": self.temp,
            "humidity": self.humidity,
            "crop_type": self.crop_type,
            "storage_method": self.storage_method
        }

    def _get_current_risk(self) -> float:
        base_risk = self.state_info[tuple(self.agent_pos)]["base_risk"][0]
        temp_effect = max(0, self.temp - 25) * 0.01
        humidity_effect = max(0, self.humidity - 60) * 0.015
        crop_factor = [0.8, 1.0, 1.2][self.crop_type]
        storage_effect = [0.0, 0.02, 0.05][self.storage_method]
        return np.clip(base_risk + temp_effect + humidity_effect + storage_effect * crop_factor, 0, 1)

    def _get_recommended_action(self) -> str:
        current_risk = self._get_current_risk()
        if current_risk > 0.7:
            return "Apply Treatment immediately"
        elif current_risk > 0.5:
            return "Move to safer area or apply treatment"
        return "Monitor conditions"

    def _flatten_state(self) -> np.ndarray:
        return np.array([
            self.current_state["position"][0],
            self.current_state["position"][1],
            self.current_state["risk_level"][0],
            self.current_state["temp"],
            self.current_state["humidity"],
            self.current_state["crop_type"],
            self.current_state["storage_method"]
        ], dtype=np.float32)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == 'human' and self.visualizer:
            return self.visualizer.render()
        elif self.render_mode == 'console':
            print(
                f"Day {self.day}: Pos {tuple(self.agent_pos)} | "
                f"Risk: {self.current_state['risk_level'][0]:.2f} | "
                f"Temp: {self.temp:.1f}°C | Hum: {self.humidity:.1f}%"
            )
        elif self.render_mode == 'rgb_array':
            if self.visualizer:
                return self.visualizer.get_rgb_array()
            return np.zeros((600, 800, 3), dtype=np.uint8)
        return None

    def close(self):
        if self.visualizer:
            self.visualizer.close()


class StorageVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.size = 600
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("Storage Environment Visualization")
        self.clock = pygame.time.Clock()
        self.grid_size = env.grid_size

    def render(self):
        self.screen.fill((255, 255, 255))  # White background

        cell_size = self.size // self.grid_size

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw agent
        agent_x, agent_y = self.env.agent_pos
        agent_rect = pygame.Rect(agent_x * cell_size, agent_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (0, 128, 255), agent_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def get_rgb_array(self):
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))  # swap axes for proper image orientation

    def close(self):
        pygame.quit()

    def reset(self):
        self.render()