import random
import requests
from typing import Tuple
import os
from datetime import datetime
import numpy as np

class WeatherSimulator:
    """
    Complete weather system with:
    - Real API fallback to mock data
    - Seasonal/daily variations
    - Location-based presets
    - Caching for performance
    """
    
    # Climate profiles for different regions
    CLIMATE_PROFILES = {
        "Kigali": {
            "temp_range": (18, 28),  # Min/Max annual temps
            "humidity_range": (60, 80),
            "rainy_seasons": [(9, 11), (3, 5)]  # Month ranges
        },
        "Nairobi": {
            "temp_range": (12, 25),
            "humidity_range": (50, 70),
            "rainy_seasons": [(3, 5), (10, 12)]
        },
        "Default": {
            "temp_range": (10, 35),
            "humidity_range": (30, 90),
            "rainy_seasons": []
        }
    }

    def __init__(self, use_real_api=False, location="Kigali"):
        self.use_real_api = use_real_api
        self.location = location
        self.last_update = None
        self.cached_data = None
        self.profile = self.CLIMATE_PROFILES.get(location, self.CLIMATE_PROFILES["Default"])

    def get_weather(self) -> Tuple[float, float]:
        """Main interface matching custom_env.py requirements"""
        if self.use_real_api and self._api_available():
            try:
                return self._get_api_weather()
            except Exception:
                # silently fallback on failure
                pass
        return self._get_simulated_weather()

    def _get_api_weather(self) -> Tuple[float, float]:
        """Fetch from OpenWeatherMap API"""
        api_key = os.getenv("OWM_API_KEY", "your_api_key_here")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=3)
        data = response.json()
        self.cached_data = (data["main"]["temp"], data["main"]["humidity"])
        self.last_update = datetime.now()
        return self.cached_data

    def _get_simulated_weather(self) -> Tuple[float, float]:
        """Generate realistic mock data"""
        current_month = datetime.now().month
        is_rainy = any(start <= current_month <= end 
                       for start, end in self.profile["rainy_seasons"])
        
        # Base temperature with seasonal variation
        temp_min, temp_max = self.profile["temp_range"]
        base_temp = temp_min + (temp_max - temp_min) * (
            0.5 + 0.3 * np.cos(2 * np.pi * (current_month - 6) / 12))
        
        # Add daily fluctuation
        temp = base_temp + random.uniform(-3, 3)
        
        # Humidity based on season and randomness
        hum_min, hum_max = self.profile["humidity_range"]
        humidity = hum_min + (hum_max - hum_min) * (
            0.7 if is_rainy else 0.3) + random.uniform(-10, 10)
        
        # Clip and ensure realistic bounds
        temp_clipped = max(min(temp, temp_max + 5), temp_min - 5)
        humidity_clipped = max(min(humidity, 95), 25)
        
        return (temp_clipped, humidity_clipped)

    def _api_available(self) -> bool:
        """Check if API should be used"""
        if self.last_update and (datetime.now() - self.last_update).seconds < 3600:
            return False  # Don't call API more than once per hour
        return True


# Singleton instance for easy import
_weather_sim = WeatherSimulator()

# Interface function matching custom_env.py requirements
def get_weather(location: str) -> Tuple[float, float]:
    """Public function that custom_env.py calls"""
    # Update simulator location on call to keep location synced
    _weather_sim.location = location
    return _weather_sim.get_weather()