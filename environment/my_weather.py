import random
import requests
from typing import Tuple
import os
from datetime import datetime
import numpy as np

class WeatherSimulator:
    """
    Complete weather simulation system with:
    - Real API fallback to mock data
    - Grid position-based weather effects
    - Seasonal and daily variations
    - Climate profiles for different regions
    """

    # Detailed climate profiles with grid effects
    CLIMATE_PROFILES = {
        "Kigali": {
            "temp_range": (18, 28),  # Min/Max annual temps (°C)
            "humidity_range": (60, 80),  # Min/Max humidity (%)
            "rainy_seasons": [(9, 11), (3, 5)],  # Month ranges (1-12)
            "grid_effects": {
                (0, 0): {"temp_mod": +4, "humidity_mod": +20},
                (2, 2): {"temp_mod": 0, "humidity_mod": 0},
                (1, 1): {"temp_mod": +1, "humidity_mod": +10},
                (3, 3): {"temp_mod": +2, "humidity_mod": -5}
            }
        },
        "Nairobi": {
            "temp_range": (12, 25),
            "humidity_range": (50, 70),
            "rainy_seasons": [(3, 5), (10, 12)],
            "grid_effects": {
                (0, 0): {"temp_mod": +3, "humidity_mod": +15},
                (2, 2): {"temp_mod": -1, "humidity_mod": -5}
            }
        },
        "Default": {
            "temp_range": (10, 35),
            "humidity_range": (30, 90),
            "rainy_seasons": [],
            "grid_effects": {}
        }
    }

    def __init__(self, use_real_api: bool = False, location: str = "Kigali"):
        self.use_real_api = use_real_api
        self.location = location
        self.profile = self.CLIMATE_PROFILES.get(location, self.CLIMATE_PROFILES["Default"])
        self.last_api_call = None
        self.cache_expiry = 3600  # 1 hour

    def get_weather(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Get weather data for specific grid position
        """
        if self.use_real_api and self._should_call_api():
            try:
                base_temp, base_humidity = self._get_api_weather()
                return self._apply_grid_effects(grid_pos, base_temp, base_humidity)
            except Exception as e:
                print(f"API call failed: {e}. Using simulated data.")
                return self._get_simulated_weather(grid_pos)

        return self._get_simulated_weather(grid_pos)

    def _get_api_weather(self) -> Tuple[float, float]:
        api_key = os.getenv("OWM_API_KEY", "your_api_key_here")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={api_key}&units=metric"

        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()

        self.last_api_call = datetime.now()
        return data["main"]["temp"], data["main"]["humidity"]

    def _get_simulated_weather(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        current_month = datetime.now().month
        is_rainy = self._is_rainy_season(current_month)

        temp_min, temp_max = self.profile["temp_range"]
        seasonal_temp = self._calculate_seasonal_temp(current_month, temp_min, temp_max)

        hum_min, hum_max = self.profile["humidity_range"]
        base_humidity = self._calculate_base_humidity(is_rainy, hum_min, hum_max)

        daily_temp = seasonal_temp + random.uniform(-3, 3)
        daily_humidity = base_humidity + random.uniform(-10, 10)

        return self._apply_grid_effects(grid_pos, daily_temp, daily_humidity)

    def _apply_grid_effects(self, grid_pos: Tuple[int, int], temp: float, humidity: float) -> Tuple[float, float]:
        effects = self.profile["grid_effects"].get(grid_pos, {})
        modified_temp = temp + effects.get("temp_mod", 0)
        modified_humidity = humidity + effects.get("humidity_mod", 0)

        return (
            np.clip(modified_temp, 10, 40),
            np.clip(modified_humidity, 25, 95)
        )

    def _calculate_seasonal_temp(self, month: int, temp_min: float, temp_max: float) -> float:
        return temp_min + (temp_max - temp_min) * (
            0.5 + 0.3 * np.cos(2 * np.pi * (month - 6) / 12))

    def _calculate_base_humidity(self, is_rainy: bool, hum_min: float, hum_max: float) -> float:
        return hum_min + (hum_max - hum_min) * (0.7 if is_rainy else 0.3)

    def _is_rainy_season(self, month: int) -> bool:
        return any(start <= month <= end for start, end in self.profile["rainy_seasons"])

    def _should_call_api(self) -> bool:
        if not self.last_api_call:
            return True
        return (datetime.now() - self.last_api_call).seconds > self.cache_expiry


# Singleton instance
_weather_simulator = WeatherSimulator()

def get_weather(location: str, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
    """
    Public function for external modules to get weather
    
    Always returns a valid (temp, humidity) tuple.
    """
    try:
        _weather_simulator.location = location
        return _weather_simulator.get_weather(grid_pos)
    except Exception as e:
        print(f"Weather system failed: {e}. Returning fallback.")
        return 25.0, 70.0  # Reasonable default values