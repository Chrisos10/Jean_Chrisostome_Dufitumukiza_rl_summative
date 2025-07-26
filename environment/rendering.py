import pygame
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime

class StorageVisualizer:
    def __init__(self, env, record_gif: bool = False):
        """
        PyGame visualization for StorageEnv
        
        Args:
            env: The StorageEnv instance to visualize
            record_gif: Whether to save frames for GIF creation
        """
        self.env = env
        self.record_gif = record_gif
        self.frames = [] if record_gif else None
        
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Farm Storage Optimization")
        self.clock = pygame.time.Clock()
        
        # Load assets
        self._load_assets()
        
        # GIF recording
        self.recording_dir = "recordings"
        os.makedirs(self.recording_dir, exist_ok=True)

    def reset(self):
        """Reset any internal state if needed (e.g., clear frames)"""
        if self.record_gif:
            self.frames = []

    def _load_assets(self):
        """Initialize visualization assets"""
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24)
        
        # Color scheme
        self.colors = {
            'background': (240, 240, 240),
            'safe': (100, 200, 100),
            'warning': (255, 200, 100),
            'danger': (220, 100, 100),
            'text': (50, 50, 50),
            'action': (70, 70, 200),
            'frame': (200, 200, 200)
        }
        
        # Icons (simplified with shapes)
        self.icons = {
            'crop': self._create_crop_icon(),
            'silo': self._create_silo_icon(),
            'ventilation': self._create_ventilation_icon()
        }

    def _create_crop_icon(self) -> pygame.Surface:
        """Create a simple crop icon"""
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(surf, (100, 180, 100), (20, 20), 18)
        return surf

    def _create_silo_icon(self) -> pygame.Surface:
        """Create a simple storage icon"""
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.rect(surf, (150, 150, 150), (5, 10, 30, 30))
        return surf

    def _create_ventilation_icon(self) -> pygame.Surface:
        """Create a ventilation icon"""
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200, 200, 200), (20, 20), 15)
        for i in range(4):
            angle = i * 90
            x = 20 + 20 * np.cos(np.radians(angle))
            y = 20 + 20 * np.sin(np.radians(angle))
            pygame.draw.line(surf, (100, 100, 100), (20, 20), (x, y), 2)
        return surf

    def render(self) -> Optional[np.ndarray]:
        """Render the current environment state"""
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw main storage visualization
        self._draw_storage_facility()
        
        # Draw status panel
        self._draw_status_panel()
        
        # Draw action history
        self._draw_action_history()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # Control rendering speed
        
        # Capture frame if recording
        if self.record_gif:
            self._capture_frame()
        
        return self._get_rgb_array()

    def _draw_storage_facility(self):
        """Draw the main storage visualization"""
        # Draw facility frame
        pygame.draw.rect(self.screen, self.colors['frame'], (200, 100, 400, 350), border_radius=10)
        
        # Draw crop storage area (color based on pest level)
        pest_color = self._get_pest_color()
        pygame.draw.rect(self.screen, pest_color, (250, 150, 300, 250), border_radius=5)
        
        # Draw icons
        self.screen.blit(self.icons['crop'], (360, 260))
        self.screen.blit(self.icons['silo'], (700, 30))
        
        # Draw pest level indicator
        self._draw_pest_meter(650, 150)

    def _draw_pest_meter(self, x: int, y: int):
        """Draw visual pest level indicator"""
        # Background
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y, 30, 200))
        
        # Fill level
        fill_height = int(200 * self.env.pest_level)
        pest_color = self._get_pest_color()
        pygame.draw.rect(self.screen, pest_color, (x, y + (200 - fill_height), 30, fill_height))
        
        # Markers
        for i in range(0, 201, 40):
            pygame.draw.line(self.screen, (100, 100, 100), (x - 5, y + i), (x, y + i), 2)
            if i < 200:
                value_text = self.font_small.render(f"{100 - i//2}%", True, self.colors['text'])
                self.screen.blit(value_text, (x - 40, y + i - 10))

    def _draw_status_panel(self):
        """Draw the right-side status panel"""
        # Panel background
        pygame.draw.rect(self.screen, (230, 230, 230), (600, 400, 180, 180), border_radius=5)
        
        # Status text
        texts = [
            f"Day: {self.env.day}/{self.env.config['max_days']}",
            f"Temp: {self.env.state['temp'][0]:.1f}°C",
            f"Humidity: {self.env.state['humidity'][0]:.1f}%",
            f"Crop: {self.env.CROP_TYPES[self.env.state['crop_type']]}",
            f"Storage: {self.env.STORAGE_METHODS[self.env.state['storage_method']]}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (610, 410 + i * 25))

    def _draw_action_history(self):
        """Draw the action history at the bottom"""
        if hasattr(self.env, 'last_action') and self.env.last_action is not None:
            action_text = f"Last Action: {self.env.ACTIONS[self.env.last_action]}"
            text_surface = self.font_large.render(action_text, True, self.colors['action'])
            self.screen.blit(text_surface, (50, 500))

    def _get_pest_color(self) -> Tuple[int, int, int]:
        """Get color based on current pest level"""
        if self.env.pest_level < 0.3:
            return self.colors['safe']
        elif self.env.pest_level < 0.7:
            return self.colors['warning']
        else:
            return self.colors['danger']

    def _capture_frame(self):
        """Capture current frame for GIF recording"""
        rgb_array = self._get_rgb_array()
        if rgb_array is not None:
            self.frames.append(rgb_array)

    def _get_rgb_array(self) -> Optional[np.ndarray]:
        """Get current frame as RGB array for video recording"""
        try:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )
        except:
            return None

    def get_rgb_array(self) -> Optional[np.ndarray]:
        """Public accessor for RGB array"""
        return self._get_rgb_array()

    def save_gif(self, filename: Optional[str] = None):
        """Save recorded frames as GIF (requires imageio)"""
        if not self.frames:
            return
            
        try:
            import imageio
            filename = filename or f"{self.recording_dir}/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            imageio.mimsave(filename, self.frames, fps=5)
            print(f"Saved GIF to {filename}")
        except ImportError:
            print("GIF saving requires imageio package")

    def close(self):
        """Clean up resources"""
        if self.record_gif and self.frames:
            self.save_gif()
        pygame.quit()