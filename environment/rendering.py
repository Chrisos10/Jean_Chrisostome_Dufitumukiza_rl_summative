import pygame
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime

class StorageVisualizer:
    """Enhanced visualization with grid system and recommendations"""
    
    def __init__(self, env, record_gif: bool = False):
        self.env = env
        self.record_gif = record_gif
        self.frames = [] if record_gif else None
        
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Farm Storage Optimization")
        self.clock = pygame.time.Clock()
        
        self._load_assets()
        self.recording_dir = "recordings"
        os.makedirs(self.recording_dir, exist_ok=True)

    def reset(self):
        """Reset visualization"""
        if self.record_gif:
            self.frames = []

    def _load_assets(self):
        """Initialize visualization assets"""
        self.font_small = pygame.font.SysFont('Arial', 16)
        self.font_medium = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 22)
        self.font_tiny = pygame.font.SysFont('Arial', 14)
        
        self.colors = {
            'background': (240, 240, 240),
            'safe': (100, 200, 100),
            'warning': (255, 200, 100),
            'danger': (220, 100, 100),
            'text': (50, 50, 50),
            'action': (70, 70, 200),
            'recommendation': (0, 150, 255),
            'grid_line': (180, 180, 180),
            'position': (0, 0, 0),
            'frame': (200, 200, 200),
            'legend': (80, 80, 80)
        }
        
        self.icons = {
            'crop': self._create_crop_icon(),
            'silo': self._create_silo_icon(),
            'ventilation': self._create_ventilation_icon(),
            'position': self._create_position_icon()
        }

    def _create_crop_icon(self):
        """More subtle crop icon"""
        surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(surf, (80, 160, 80, 150), (15, 15), 12)  # Semi-transparent
        return surf

    def _create_silo_icon(self):
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.rect(surf, (150, 150, 150), (5, 10, 30, 30))
        return surf

    def _create_ventilation_icon(self):
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200, 200, 200), (20, 20), 15)
        for i in range(4):
            angle = i * 90
            x = 20 + 20 * np.cos(np.radians(angle))
            y = 20 + 20 * np.sin(np.radians(angle))
            pygame.draw.line(surf, (100, 100, 100), (20, 20), (x, y), 2)
        return surf

    def _create_position_icon(self):
        surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(surf, self.colors['position'], (15, 15), 12, 3)
        return surf

    def render(self) -> Optional[np.ndarray]:
        """Render current environment state"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        self.screen.fill(self.colors['background'])
        
        # Draw main components
        self._draw_storage_facility()
        self._draw_status_panel()
        self._draw_action_history()
        self._draw_legend()
        
        pygame.display.flip()
        self.clock.tick(10)
        
        if self.record_gif:
            self._capture_frame()
        
        return self._get_rgb_array()

    def _draw_storage_facility(self):
        """Draw the main storage visualization with grid"""
        # Facility frame
        pygame.draw.rect(self.screen, self.colors['frame'], (200, 100, 400, 350), border_radius=10)
        
        # Draw the grid
        self._draw_grid()
        
        # Pest level overlay
        pest_color = self._get_pest_color()
        s = pygame.Surface((300, 200), pygame.SRCALPHA)
        s.fill((*pest_color[:3], 100))
        self.screen.blit(s, (250, 150))
        
        # Draw icons
        self.screen.blit(self.icons['silo'], (700, 30))
        
        # Pest meter
        self._draw_pest_meter(650, 150)

    def _draw_grid(self):
        """Draw the storage grid system"""
        grid_width, grid_height = 300, 200
        start_x, start_y = 250, 150
        cell_width = grid_width // self.env.grid_size[1]
        cell_height = grid_height // self.env.grid_size[0]
        
        for row in range(self.env.grid_size[0]):
            for col in range(self.env.grid_size[1]):
                x = start_x + col * cell_width
                y = start_y + row * cell_height
                
                # Cell color based on zone risk
                risk = self.env._get_zone_risk([row, col])
                color = (
                    int(255 * (1 - risk*0.7)),
                    int(255 * (1 - risk*0.7)),
                    100
                )
                pygame.draw.rect(self.screen, color, (x, y, cell_width, cell_height))
                pygame.draw.rect(self.screen, self.colors['grid_line'], (x, y, cell_width, cell_height), 1)
                
                # Current position marker
                if [row, col] == self.env.current_pos:
                    self.screen.blit(self.icons['position'], 
                                   (x + cell_width//2 - 15, y + cell_height//2 - 15))

    def _draw_pest_meter(self, x: int, y: int):
        """Draw pest level indicator"""
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
        """Improved right-side panel without overlapping"""
        panel_width = 200
        pygame.draw.rect(self.screen, (240, 240, 240), (600, 400, panel_width, 180), border_radius=5)
        
        # Column 1
        texts = [
            f"Day: {self.env.day}/{self.env.config['max_days']}",
            f"Position: {self.env.current_pos}",
            f"Temp: {self.env.state['temp'][0]:.1f}°C",
            f"Humidity: {self.env.state['humidity'][0]:.1f}%"
        ]
        for i, text in enumerate(texts):
            self.screen.blit(
                self.font_small.render(text, True, self.colors['text']),
                (610, 410 + i * 25)
            )
        
        # Column 2
        texts = [
            f"Crop: {self.env.CROP_TYPES[self.env.state['crop_type']]}",
            f"Storage: {self.env.STORAGE_METHODS[self.env.state['storage_method']]}"
        ]
        for i, text in enumerate(texts):
            self.screen.blit(
                self.font_small.render(text, True, self.colors['text']),
                (610 + panel_width//2, 410 + i * 25)
            )
        
        # Recommended action
        if hasattr(self.env, 'last_info'):
            rec_text = f"Suggested: {self.env.last_info['recommended']}"
            self.screen.blit(
                self.font_medium.render(rec_text, True, self.colors['recommendation']),
                (610, 520)
            )

    def _draw_action_history(self):
        """Show current action at bottom"""
        if hasattr(self.env, 'last_action'):
            action_text = f"Current Action: {self.env.ACTIONS[self.env.last_action]}"
            text_surface = self.font_large.render(action_text, True, self.colors['action'])
            self.screen.blit(text_surface, (50, 500))

    def _draw_legend(self):
        """Add color legend for grid zones"""
        legend_texts = [
            "Zone Colors:",
            "Green = Low Risk",
            "Yellow = Medium Risk",
            "Red = High Risk"
        ]
        
        for i, text in enumerate(legend_texts):
            color = self.colors['legend'] if i == 0 else self.colors['text']
            text_surf = self.font_tiny.render(text, True, color)
            self.screen.blit(text_surf, (250, 370 + i * 16))

    def _get_pest_color(self) -> Tuple[int, int, int]:
        """Get color based on pest level"""
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
        """Get current frame as RGB array"""
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
        """Save recorded frames as GIF"""
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