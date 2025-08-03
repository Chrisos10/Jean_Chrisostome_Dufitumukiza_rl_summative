import pygame
import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime
import textwrap

class StorageVisualizer:
    """Advanced visualizer with professional styling and responsive layout"""

    ZONE_COLORS = [
        (100, 255, 100),  # Optimal Storage
        (100, 200, 255),  # Too Dry
        (200, 255, 200),  # Well Ventilated
        (255, 255, 100),  # High Humidity
        (255, 80, 80)     # Critical Risk
    ]

    def __init__(self, env, record_gif: bool = False):
        self.env = env
        self.record_gif = record_gif
        self.frames = [] if record_gif else None
        self.agent_path = []
        self.last_direction = None
        self.animation_frame = 0
        
        # Track episode rewards for progress visualization
        self.episode_total_reward = 0.0
        
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((1000, 750))
            pygame.display.set_caption("Farm Storage Optimization")
        except pygame.error as e:
            raise RuntimeError(f"Display initialization failed: {str(e)}")
        
        self.clock = pygame.time.Clock()
        self._init_fonts()
        self._init_colors()
        self._calculate_layout()

    def _init_fonts(self):
        pygame.font.init()
        try:
            self.font_title = pygame.font.SysFont('Arial', 20, bold=True)
            self.font_large = pygame.font.SysFont('Arial', 16, bold=True)
            self.font_medium = pygame.font.SysFont('Arial', 14)
            self.font_small = pygame.font.SysFont('Arial', 12)
            self.font_tiny = pygame.font.SysFont('Arial', 10)
        except:
            self.font_title = pygame.font.SysFont(None, 20)
            self.font_large = pygame.font.SysFont(None, 16)
            self.font_medium = pygame.font.SysFont(None, 14)
            self.font_small = pygame.font.SysFont(None, 12)
            self.font_tiny = pygame.font.SysFont(None, 10)

    def _init_colors(self):
        self.ui_colors = {
            'background': (245, 245, 250),
            'panel_bg': (255, 255, 255),
            'panel_border': (200, 200, 210),
            'text_primary': (40, 40, 50),
            'text_secondary': (80, 80, 90),
            'text_accent': (0, 120, 200),
            'success': (40, 180, 99),
            'warning': (243, 156, 18),
            'danger': (231, 76, 60),
            'agent_primary': (52, 152, 219),
            'agent_secondary': (241, 196, 15),
            'pest_low': (46, 204, 113),
            'pest_medium': (230, 126, 34),
            'pest_high': (231, 76, 60),
            'empty_cell': (220, 220, 230),
            'phase_analyze': (52, 152, 219),
            'phase_navigate': (243, 156, 18),
            'phase_treat': (40, 180, 99),
            'reward_positive': (46, 204, 113),
            'reward_negative': (231, 76, 60),
            'reward_neutral': (149, 165, 166),
            'curriculum_easy': (46, 204, 113),
            'curriculum_medium': (243, 156, 18),
            'curriculum_hard': (231, 76, 60)
        }

    def _calculate_layout(self):
        screen_width, screen_height = self.screen.get_size()
        self.title_area = {'x': 0, 'y': 0, 'width': screen_width, 'height': 70}
        self.legend_area = {'x': 30, 'y': self.title_area['height'] + 5, 'width': screen_width - 60, 'height': 35}
        self.grid_area = {'x': 30, 'y': self.legend_area['y'] + self.legend_area['height'] + 10, 'width': 500, 'height': 500}
        self.panel_area = {'x': 550, 'y': self.grid_area['y'], 'width': 420, 'height': 500}
        self.recommendation_area = {'x': 30, 'y': self.grid_area['y'] + self.grid_area['height'] + 15, 'width': 940, 'height': 80}
        self.cell_size = min(
            self.grid_area['width'] // self.env.grid_size[1],
            self.grid_area['height'] // self.env.grid_size[0]
        )

    def reset(self):
        self.agent_path = []
        self.last_direction = None
        self.animation_frame = 0
        # Reset episode reward tracking
        self.episode_total_reward = 0.0
        if self.record_gif:
            self.frames = []

    def update_reward(self, reward: float):
        """Update the total episode reward"""
        self.episode_total_reward += reward

    def render(self) -> Optional[np.ndarray]:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Window closed")

            self.animation_frame += 1
            self.screen.fill(self.ui_colors['background'])
            self._draw_title_bar()
            self._draw_legend()
            self._draw_zone_grid()
            self._draw_agent_with_effects()
            self._draw_information_panel()
            self._draw_recommendation_panel()
            pygame.display.flip()
            self.clock.tick(30)
            if self.record_gif:
                self._capture_frame()
            return self._get_rgb_array()
        except Exception as e:
            print(f"Rendering error: {str(e)}")
            return None

    def _draw_title_bar(self):
        title_area = self.title_area
        pygame.draw.rect(self.screen, self.ui_colors['panel_bg'], (title_area['x'], title_area['y'], title_area['width'], title_area['height']), border_radius=0)
        pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (0, title_area['height'] - 2, title_area['width'], 2))
        
        # Main title with episode phase
        phase_colors = {
            'ANALYZE': self.ui_colors['phase_analyze'],
            'NAVIGATE': self.ui_colors['phase_navigate'], 
            'TREAT': self.ui_colors['phase_treat']
        }
        phase_color = phase_colors.get(self.env.episode_phase, self.ui_colors['text_primary'])
        
        title = f"Farm Storage - Episode {self.env.step_count} | Phase: {self.env.episode_phase}"
        title_surface = self.font_title.render(title, True, phase_color)
        self.screen.blit(title_surface, (20, 15))
        
        # Episode Progress (Total Reward)
        reward_color = self._get_reward_color(self.episode_total_reward)
        reward_text = f"Episode Reward: {self.episode_total_reward:.1f}"
        reward_surface = self.font_medium.render(reward_text, True, reward_color)
        self.screen.blit(reward_surface, (20, 40))
        
        # Curriculum Stage Indicator
        curriculum_stage = self.env.config.get('curriculum_stage', 1)
        stage_colors = {
            1: self.ui_colors['curriculum_easy'],
            2: self.ui_colors['curriculum_medium'], 
            3: self.ui_colors['curriculum_hard']
        }
        stage_names = {1: "Easy", 2: "Medium", 3: "Hard"}
        stage_color = stage_colors.get(curriculum_stage, self.ui_colors['text_secondary'])
        stage_text = f"Curriculum: Stage {curriculum_stage} ({stage_names.get(curriculum_stage, 'Unknown')})"
        stage_surface = self.font_medium.render(stage_text, True, stage_color)
        # Position it on the right side of the title bar
        self.screen.blit(stage_surface, (title_area['width'] - stage_surface.get_width() - 20, 15))
        
        # Status metrics
        pest_color = self._get_pest_color(self.env.pest_level)
        metrics = [
            f"Position: [{self.env.current_pos[0]}, {self.env.current_pos[1]}]",
            f"Pest: {self.env.pest_level*100:.1f}%",
            f"Temp: {self.env.temp:.1f}°C",  
            f"Humidity: {self.env.humidity:.1f}%"
        ]
        x_offset = self.screen.get_width() - 480
        for i, metric in enumerate(metrics):
            if "Pest:" in metric:
                color = pest_color
            elif "Temp:" in metric:
                color = self._get_temp_color(self.env.temp)
            elif "Humidity:" in metric:
                color = self._get_humidity_color(self.env.humidity)
            else:
                color = self.ui_colors['text_secondary']
            metric_surface = self.font_small.render(metric, True, color)
            self.screen.blit(metric_surface, (x_offset + i * 115, 40))

    def _get_reward_color(self, reward: float) -> tuple:
        """Get color based on reward value"""
        if reward > 0:
            return self.ui_colors['reward_positive']
        elif reward < 0:
            return self.ui_colors['reward_negative']
        else:
            return self.ui_colors['reward_neutral']

    def _draw_legend(self):
        legend_area = self.legend_area
        pygame.draw.rect(self.screen, self.ui_colors['panel_bg'], (legend_area['x'], legend_area['y'], legend_area['width'], legend_area['height']), border_radius=8)
        pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (legend_area['x'], legend_area['y'], legend_area['width'], legend_area['height']), 1, border_radius=8)
        legend_title = self.font_small.render("Zone Types:", True, self.ui_colors['text_accent'])
        self.screen.blit(legend_title, (legend_area['x'] + 10, legend_area['y'] + 5))
        
        x_offset = legend_area['x'] + 100
        zone_spacing = 150
        for i, zone_type in enumerate([0, 1, 2, 3, 4]):
            zone = self.env.STATE_TYPES[zone_type]
            swatch_x = x_offset + i * zone_spacing
            swatch_y = legend_area['y'] + 8
            pygame.draw.rect(self.screen, self.ZONE_COLORS[zone_type], (swatch_x, swatch_y, 18, 18), border_radius=4)
            pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (swatch_x, swatch_y, 18, 18), 1, border_radius=4)
            
            short_names = {
                "Optimal Storage": "Optimal",
                "Too Dry": "Too Dry", 
                "Well Ventilated": "Ventilated",
                "High Humidity": "High Humid",
                "Critical Risk": "Critical"
            }
            label_text = short_names.get(zone["name"], zone["name"])
            label_surface = self.font_tiny.render(label_text, True, self.ui_colors['text_secondary'])
            self.screen.blit(label_surface, (swatch_x + 25, swatch_y + 2))

    def _draw_zone_grid(self):
        grid_x, grid_y = self.grid_area['x'], self.grid_area['y']
        shadow_offset = 3
        pygame.draw.rect(self.screen, (0, 0, 0, 30), (grid_x + shadow_offset, grid_y + shadow_offset, self.env.grid_size[1] * self.cell_size, self.env.grid_size[0] * self.cell_size), border_radius=8)
        
        for row in range(self.env.grid_size[0]):
            for col in range(self.env.grid_size[1]):
                zone_type = self.env.grid_states[row, col]
                x = grid_x + col * self.cell_size
                y = grid_y + row * self.cell_size
                
                if zone_type == -1:
                    # Draw empty cell
                    pygame.draw.rect(
                        self.screen,
                        self.ui_colors['empty_cell'],
                        (x, y, self.cell_size, self.cell_size),
                        border_radius=6
                    )
                    pygame.draw.rect(
                        self.screen,
                        self.ui_colors['panel_border'],
                        (x, y, self.cell_size, self.cell_size),
                        1, border_radius=6
                    )
                    continue
                
                # Draw zone with highlight if it's the target zone
                color = self.ZONE_COLORS[zone_type]
                
                # Highlight target zone if conditions have been analyzed
                if (self.env.has_read_conditions and 
                    zone_type == self.env.target_zone):
                    # Add golden border for target zone
                    pygame.draw.rect(self.screen, (255, 215, 0), (x-2, y-2, self.cell_size+4, self.cell_size+4), border_radius=8)
                
                # Draw zone with 3D effect
                for i in range(3):
                    shade = tuple(max(0, c - i * 10) for c in color)
                    pygame.draw.rect(self.screen, shade, (x + i, y + i, self.cell_size - 2*i, self.cell_size - 2*i), border_radius=6 - i)
                
                # Zone label
                zone_name = self.env.STATE_TYPES[zone_type]["name"]
                short_name = self._get_short_zone_name(zone_name)
                text_surface = self.font_small.render(short_name, True, (255, 255, 255))
                text_bg = pygame.Surface((text_surface.get_width() + 6, text_surface.get_height() + 4))
                text_bg.fill((0, 0, 0))
                text_bg.set_alpha(120)
                text_x = x + self.cell_size//2 - text_surface.get_width()//2
                text_y = y + self.cell_size//2 - text_surface.get_height()//2
                self.screen.blit(text_bg, (text_x - 3, text_y - 2))
                self.screen.blit(text_surface, (text_x, text_y))
                pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (x, y, self.cell_size, self.cell_size), 2, border_radius=6)

    def _get_short_zone_name(self, name):
        name_map = {
            "Optimal Storage": "Opt",
            "Too Dry": "Dry",
            "Well Ventilated": "Vent",
            "High Humidity": "Humid",
            "Critical Risk": "RISK"
        }
        return name_map.get(name, name[:4])

    def _draw_agent_with_effects(self):
        grid_x, grid_y = self.grid_area['x'], self.grid_area['y']
        agent_x = grid_x + self.env.current_pos[1] * self.cell_size + self.cell_size//2
        agent_y = grid_y + self.env.current_pos[0] * self.cell_size + self.cell_size//2
        
        # Animated pulse effect
        pulse = abs(np.sin(self.animation_frame * 0.2)) * 3
        
        # Shadow
        pygame.draw.circle(self.screen, (0, 0, 0, 50), (agent_x + 2, agent_y + 2), int(15 + pulse))
        
        # Agent body with phase-based color
        phase_colors = {
            'ANALYZE': self.ui_colors['phase_analyze'],
            'NAVIGATE': self.ui_colors['phase_navigate'],
            'TREAT': self.ui_colors['phase_treat']
        }
        agent_color = phase_colors.get(self.env.episode_phase, self.ui_colors['agent_primary'])
        
        for i in range(3):
            radius = int(15 + pulse - i * 2)
            color_intensity = 255 - i * 40
            current_color = (
                min(255, agent_color[0] + color_intensity - 200),
                min(255, agent_color[1] + color_intensity - 200),
                min(255, agent_color[2] + color_intensity - 100)
            )
            pygame.draw.circle(self.screen, current_color, (agent_x, agent_y), radius)

    def _draw_information_panel(self):
        panel_x, panel_y = self.panel_area['x'], self.panel_area['y']
        panel_w, panel_h = self.panel_area['width'], self.panel_area['height']
        
        # Panel background with shadow
        pygame.draw.rect(self.screen, (0, 0, 0, 20), (panel_x + 3, panel_y + 3, panel_w, panel_h), border_radius=12)
        pygame.draw.rect(self.screen, self.ui_colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h), border_radius=12)
        pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (panel_x, panel_y, panel_w, panel_h), 2, border_radius=12)
        
        y_pos = panel_y + 20
        
        # Episode Phase Information
        self._draw_info_section("Episode Phase", panel_x + 20, y_pos)
        y_pos += 30
        
        phase_info = {
            'ANALYZE': "Read environmental conditions first (Action 21)",
            'NAVIGATE': "Move to the correct zone for current conditions",
            'TREAT': "Apply appropriate treatment for this zone"
        }
        
        phase_color = {
            'ANALYZE': self.ui_colors['phase_analyze'],
            'NAVIGATE': self.ui_colors['phase_navigate'],
            'TREAT': self.ui_colors['phase_treat']
        }
        
        phase_surface = self.font_medium.render(f"Current: {self.env.episode_phase}", True, 
                                              phase_color.get(self.env.episode_phase, self.ui_colors['text_primary']))
        self.screen.blit(phase_surface, (panel_x + 30, y_pos))
        y_pos += 25
        
        # Phase instruction
        instruction = phase_info.get(self.env.episode_phase, "")
        instruction_lines = textwrap.wrap(instruction, width=45)
        for line in instruction_lines:
            inst_surface = self.font_small.render(line, True, self.ui_colors['text_secondary'])
            self.screen.blit(inst_surface, (panel_x + 30, y_pos))
            y_pos += 18
        y_pos += 15
        
        # Current Zone Information
        zone_type = self.env.state["zone_type"]
        if zone_type == -1:
            zone_title = self.font_large.render("Current Zone: Empty", True, self.ui_colors['text_primary'])
            self.screen.blit(zone_title, (panel_x + 20, y_pos))
            y_pos += 35
            warning_surface = self.font_medium.render("No zone here. Move to a zone cell.", True, self.ui_colors['danger'])
            self.screen.blit(warning_surface, (panel_x + 20, y_pos))
            y_pos += 30
        else:
            zone = self.env.STATE_TYPES[zone_type]
            zone_title = self.font_large.render(f"Current Zone: {zone['name']}", True, self.ui_colors['text_primary'])
            self.screen.blit(zone_title, (panel_x + 20, y_pos))
            y_pos += 35
            
            # Zone description
            description_lines = textwrap.wrap(zone['description'], width=50)
            for line in description_lines:
                desc_surface = self.font_small.render(line, True, self.ui_colors['text_secondary'])
                self.screen.blit(desc_surface, (panel_x + 20, y_pos))
                y_pos += 20
            y_pos += 10
        
        # Environmental Conditions (only show if analyzed)
        if self.env.has_read_conditions:
            self._draw_info_section("Environmental Conditions", panel_x + 20, y_pos)
            y_pos += 30
            
            info_data = [
                ("Temperature", f"{self.env.temp:.1f}°C", self._get_temp_color(self.env.temp)),
                ("Humidity", f"{self.env.humidity:.1f}%", self._get_humidity_color(self.env.humidity)),
                ("Crop Type", self.env.CROP_TYPES[self.env.state['crop_type']], self.ui_colors['text_secondary']),
                ("Storage Method", self.env.STORAGE_METHODS[self.env.state['storage_method']], self.ui_colors['text_secondary']),
            ]
            
            for label, value, color in info_data:
                label_surface = self.font_medium.render(f"{label}:", True, self.ui_colors['text_secondary'])
                value_surface = self.font_medium.render(value, True, color)
                self.screen.blit(label_surface, (panel_x + 30, y_pos))
                self.screen.blit(value_surface, (panel_x + 200, y_pos))
                y_pos += 25
            y_pos += 10
            
            y_pos += 10
            
            # Pest level meter
            self._draw_pest_meter(panel_x + 30, y_pos, panel_w - 60)
        else:
            # Show that conditions need to be analyzed
            self._draw_info_section("Conditions", panel_x + 20, y_pos)
            y_pos += 35
            need_analysis = self.font_medium.render("Use action 21 to analyze conditions", True, self.ui_colors['warning'])
            self.screen.blit(need_analysis, (panel_x + 30, y_pos))

    def _draw_info_section(self, title, x, y):
        title_surface = self.font_large.render(title, True, self.ui_colors['text_accent'])
        self.screen.blit(title_surface, (x, y))
        pygame.draw.line(self.screen, self.ui_colors['text_accent'], (x, y + 22), (x + title_surface.get_width(), y + 22), 2)

    def _draw_pest_meter(self, x, y, width):
        pygame.draw.rect(self.screen, self.ui_colors['panel_border'], (x, y, width, 25), border_radius=12)
        fill_width = int(width * self.env.pest_level)
        pest_color = self._get_pest_color(self.env.pest_level)
        if fill_width > 0:
            pygame.draw.rect(self.screen, pest_color, (x, y, fill_width, 25), border_radius=12)
        pest_text = f"Pest Risk: {self.env.pest_level*100:.1f}%"
        text_color = (255, 255, 255) if self.env.pest_level > 0.5 else self.ui_colors['text_primary']
        text_surface = self.font_medium.render(pest_text, True, text_color)
        text_x = x + width//2 - text_surface.get_width()//2
        self.screen.blit(text_surface, (text_x, y + 5))

    def _draw_recommendation_panel(self):
        rec_x, rec_y = self.recommendation_area['x'], self.recommendation_area['y']
        rec_w, rec_h = self.recommendation_area['width'], self.recommendation_area['height']
        
        pygame.draw.rect(self.screen, self.ui_colors['panel_bg'], (rec_x, rec_y, rec_w, rec_h), border_radius=8)
        pygame.draw.rect(self.screen, self.ui_colors['success'], (rec_x, rec_y, rec_w, rec_h), 2, border_radius=8)
        
        # Phase-specific recommendations
        title_surface = self.font_large.render("Current Instructions:", True, self.ui_colors['success'])
        self.screen.blit(title_surface, (rec_x + 15, rec_y + 10))
        
        if self.env.episode_phase == "ANALYZE":
            instruction = "Press action 21 to read environmental conditions"
            inst_surface = self.font_medium.render(instruction, True, self.ui_colors['text_primary'])
            self.screen.blit(inst_surface, (rec_x + 15, rec_y + 35))
            
        elif self.env.episode_phase == "NAVIGATE":
            if self.env.has_read_conditions:
                instruction = f"Navigate to appropriate zone using arrow actions"
            else:
                instruction = "Read conditions first before navigating"
            inst_surface = self.font_medium.render(instruction, True, self.ui_colors['text_primary'])
            self.screen.blit(inst_surface, (rec_x + 15, rec_y + 35))
            
        elif self.env.episode_phase == "TREAT":
            zone_type = self.env.state["zone_type"]
            if zone_type != -1:
                zone_info = self.env.STATE_TYPES[zone_type]
                treatment_actions = [a for a in zone_info["actions"] if a <= 16]
                
                if treatment_actions:
                    action_names = [self.env.ACTIONS[a] for a in treatment_actions if a < len(self.env.ACTIONS)]
                    action_text = " • ".join(action_names)
                    max_chars_per_line = (rec_w - 30) // 8
                    wrapped_lines = textwrap.wrap(action_text, width=max_chars_per_line)
                    y_offset = rec_y + 35
                    for line in wrapped_lines[:2]:
                        line_surface = self.font_medium.render(line, True, self.ui_colors['text_primary'])
                        self.screen.blit(line_surface, (rec_x + 15, y_offset))
                        y_offset += 20
                else:
                    warning_surface = self.font_medium.render("No valid treatments available!", True, self.ui_colors['danger'])
                    self.screen.blit(warning_surface, (rec_x + 15, rec_y + 35))
            else:
                warning_surface = self.font_medium.render("No valid actions in empty cell!", True, self.ui_colors['danger'])
                self.screen.blit(warning_surface, (rec_x + 15, rec_y + 35))

    def _get_pest_color(self, pest_level):
        if pest_level < 0.3:
            return self.ui_colors['pest_low']
        elif pest_level < 0.7:
            return self.ui_colors['pest_medium']
        else:
            return self.ui_colors['pest_high']

    def _get_temp_color(self, temp):
        if 18 <= temp <= 25:
            return self.ui_colors['success']
        elif 15 <= temp <= 30:
            return self.ui_colors['warning']
        else:
            return self.ui_colors['danger']

    def _get_humidity_color(self, humidity):
        if 55 <= humidity <= 70:
            return self.ui_colors['success']
        elif 45 <= humidity <= 80:
            return self.ui_colors['warning']
        else:
            return self.ui_colors['danger']

    def _capture_frame(self):
        rgb_array = self._get_rgb_array()
        if rgb_array is not None:
            self.frames.append(rgb_array)

    def _get_rgb_array(self) -> Optional[np.ndarray]:
        try:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )
        except:
            return None

    def close(self):
        """Clean up resources"""
        if self.record_gif and self.frames:
            self.save_gif()
        pygame.quit()