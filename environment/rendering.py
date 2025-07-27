import pygame
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime

class StorageVisualizer:
    """Enhanced visualizer with professional agent representation"""

    # Permanent zone colors (RGB)
    ZONE_COLORS = [
        (100, 255, 100),  # Optimal (green)
        (100, 200, 255),  # Cool Dry (light blue) - will be skipped in legend
        (200, 255, 200),  # Ventilated (mint)
        (255, 255, 100),  # Protected (yellow)
        (255, 150, 100)   # Risk Zone (orange)
    ]

    def __init__(self, env, record_gif: bool = False):
        self.env = env
        self.record_gif = record_gif
        self.frames = [] if record_gif else None
        self.agent_path = []
        self.last_direction = None
        
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Farm Storage Optimization")
        except pygame.error as e:
            raise RuntimeError(f"Display initialization failed: {str(e)}")
        
        self.clock = pygame.time.Clock()
        self._init_fonts()
        self._init_colors()
        
        self.recording_dir = "recordings"
        os.makedirs(self.recording_dir, exist_ok=True)

    def _init_fonts(self):
        """Initialize font system with fallbacks"""
        pygame.font.init()
        try:
            self.font_small = pygame.font.SysFont('Arial', 14)
            self.font_medium = pygame.font.SysFont('Arial', 18)
            self.font_large = pygame.font.SysFont('Arial', 24)
        except:
            self.font_small = pygame.font.SysFont(None, 14)
            self.font_medium = pygame.font.SysFont(None, 18)
            self.font_large = pygame.font.SysFont(None, 24)

    def _init_colors(self):
        """Initialize color palette"""
        self.ui_colors = {
            'background': (240, 240, 240),
            'text': (50, 50, 50),
            'panel': (220, 220, 220),
            'recommendation': (0, 150, 0),
            'pest_meter': (200, 50, 50),
            'agent_primary': (30, 120, 200),
            'agent_secondary': (255, 215, 0)
        }

    def reset(self):
        """Reset visualization state"""
        self.agent_path = []
        self.last_direction = None
        if self.record_gif:
            self.frames = []

    def render(self) -> Optional[np.ndarray]:
        """Main rendering loop"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Window closed")

            self.screen.fill(self.ui_colors['background'])
            
            self._draw_zone_grid()
            self._draw_information_panel()
            self._draw_pest_meter()
            self._draw_action_recommendations()
            self._draw_legend()
            self._draw_agent()
            
            pygame.display.flip()
            self.clock.tick(10)
            
            if self.record_gif:
                self._capture_frame()
            
            return self._get_rgb_array()
        
        except Exception as e:
            print(f"Rendering error: {str(e)}")
            return None

    def _draw_zone_grid(self):
        """Draw the zone grid system"""
        grid_size = 400
        margin_x, margin_y = 50, 100
        cell_size = grid_size // self.env.grid_size[1]
        
        for row in range(self.env.grid_size[0]):
            for col in range(self.env.grid_size[1]):
                zone_type = self.env.grid_states[row, col]
                x = margin_x + col * cell_size
                y = margin_y + row * cell_size
                
                # Zone background
                pygame.draw.rect(
                    self.screen,
                    self.ZONE_COLORS[zone_type],
                    (x+2, y+2, cell_size-4, cell_size-4),
                    border_radius=4
                )
                
                # Zone label
                label = self.font_small.render(
                    self.env.STATE_TYPES[zone_type]["name"][:3],
                    True,
                    (0, 0, 0)
                )
                self.screen.blit(
                    label,
                    (x + cell_size//2 - label.get_width()//2,
                     y + cell_size//2 - label.get_height()//2)
                )

    def _draw_agent(self):
        """Draw a professional agent representation with direction indicator"""
        cell_size = 400 // self.env.grid_size[1]
        margin_x, margin_y = 50, 100
        
        # Current position center
        x = margin_x + self.env.current_pos[1] * cell_size + cell_size//2
        y = margin_y + self.env.current_pos[0] * cell_size + cell_size//2
        
        # Update movement direction
        if len(self.agent_path) > 1:
            prev_pos = self.agent_path[-1]
            dx = self.env.current_pos[1] - prev_pos[1]
            dy = self.env.current_pos[0] - prev_pos[0]
            self.last_direction = (dx, dy)
        
        # Agent body (blue circle with gold border)
        pygame.draw.circle(
            self.screen,
            self.ui_colors['agent_primary'],
            (x, y),
            12
        )
        pygame.draw.circle(
            self.screen,
            self.ui_colors['agent_secondary'],
            (x, y),
            12, 2
        )
        
        # Direction indicator (if moved)
        if self.last_direction:
            dx, dy = self.last_direction
            if dx != 0 or dy != 0:
                # Calculate arrow points
                if dx > 0:  # Right
                    points = [(x+8, y), (x, y-8), (x, y+8)]
                elif dx < 0:  # Left
                    points = [(x-8, y), (x, y-8), (x, y+8)]
                elif dy > 0:  # Down
                    points = [(x, y+8), (x-8, y), (x+8, y)]
                else:  # Up
                    points = [(x, y-8), (x-8, y), (x+8, y)]
                
                pygame.draw.polygon(
                    self.screen,
                    self.ui_colors['agent_secondary'],
                    points
                )

    def _draw_information_panel(self):
        """Draw the right-side information panel"""
        panel_x, panel_y = 500, 100
        zone_type = self.env.state["zone_type"]
        zone = self.env.STATE_TYPES[zone_type]
        
        # Panel background
        pygame.draw.rect(
            self.screen,
            self.ui_colors['panel'],
            (panel_x, panel_y, 250, 400),
            border_radius=8
        )
        
        # Information lines
        info_lines = [
            f"Zone: {zone['name']}",
            f"Temp: {self.env.state['temp'][0]:.1f}°C",
            f"Humidity: {self.env.state['humidity'][0]:.1f}%",
            f"Pest: {self.env.pest_level*100:.1f}%",
            "",
            f"Crop: {self.env.CROP_TYPES[self.env.state['crop_type']]}",
            f"Storage: {self.env.STORAGE_METHODS[self.env.state['storage_method']]}",
            "",
            f"Day: {self.env.day}/{self.env.config['max_days']}",
            f"Suitability: {self.env.calculate_zone_suitability()[tuple(self.env.current_pos)]:.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            if not line:
                continue
            y_pos = panel_y + 20 + i * 30
            self.screen.blit(
                self.font_medium.render(line, True, self.ui_colors['text']),
                (panel_x + 15, y_pos)
            )

    def _draw_pest_meter(self):
        """Draw the pest risk indicator"""
        pest_x, pest_y = 500, 520
        width = min(200, int(200 * self.env.pest_level))
        
        # Meter background
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            (pest_x, pest_y, 200, 20),
            border_radius=4
        )
        
        # Fill
        pygame.draw.rect(
            self.screen,
            self.ui_colors['pest_meter'],
            (pest_x, pest_y, width, 20),
            border_radius=4
        )
        
        # Text
        pest_text = self.font_medium.render(
            f"Pest Risk: {self.env.pest_level*100:.1f}%",
            True,
            self.ui_colors['text']
        )
        self.screen.blit(
            pest_text,
            (pest_x + 100 - pest_text.get_width()//2, pest_y - 25)
        )

    def _draw_action_recommendations(self):
        """Show recommended actions"""
        zone_type = self.env.state["zone_type"]
        zone = self.env.STATE_TYPES[zone_type]
        
        valid_actions = [a for a in zone["actions"] if a < 7]
        if valid_actions:
            action_text = "Recommended: " + " or ".join(
                [self.env.ACTIONS[a] for a in valid_actions]
            )
            self.screen.blit(
                self.font_medium.render(action_text, True, self.ui_colors['recommendation']),
                (50, 520)
            )

    def _draw_legend(self):
        """Draw zone color legend (excluding Cool Dry)"""
        legend_x, legend_y = 50, 30
        x_offset = 0
        
        # List of zone indices to include (excluding Cool Dry which is index 1)
        included_zones = [0, 2, 3, 4]
        
        for i in included_zones:
            zone = self.env.STATE_TYPES[i]
            # Color swatch
            pygame.draw.rect(
                self.screen,
                self.ZONE_COLORS[i],
                (legend_x + x_offset, legend_y, 20, 20),
                border_radius=3
            )
            # Label
            self.screen.blit(
                self.font_small.render(zone["name"], True, self.ui_colors['text']),
                (legend_x + x_offset + 25, legend_y)
            )
            x_offset += 120  # Space between legend items

    def _capture_frame(self):
        """Capture frame for GIF recording"""
        rgb_array = self._get_rgb_array()
        if rgb_array is not None:
            self.frames.append(rgb_array)

    def _get_rgb_array(self) -> Optional[np.ndarray]:
        """Convert screen to RGB array"""
        try:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )
        except:
            return None

    def save_gif(self, filename: Optional[str] = None):
        """Save recorded frames as GIF"""
        if not self.frames or not self.record_gif:
            return
            
        try:
            import imageio
            filename = filename or f"{self.recording_dir}/sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            imageio.mimsave(filename, self.frames, fps=5)
            print(f"Saved GIF: {filename}")
        except ImportError:
            print("GIF saving requires imageio package")

    def close(self):
        """Clean up resources"""
        if self.record_gif and self.frames:
            self.save_gif()
        pygame.quit()