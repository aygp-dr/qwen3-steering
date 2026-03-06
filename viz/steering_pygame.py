#!/usr/bin/env python3
"""
Visualization 5: Interactive Steering Vector Explorer (Pygame)

2D projection of activation space with:
- Particles representing token positions in the residual stream
- A draggable steering vector arrow (rotate and scale with mouse)
- Real-time particle displacement as the user adjusts the vector
- Color-coded regions for style basins
- Layer-by-layer animation: shows perturbation propagation through layers
- Keyboard controls:
    Arrow Up/Down: increase/decrease alpha
    Arrow Left/Right: rotate steering direction
    Space: reset to baseline
    1-4: select style axis (terse, formal, socratic, dry-wit)
    L: toggle layer animation
    Tab: cycle through layers
    Escape: quit

This is "4D in 2D+time" -- the time axis is layer progression.
"""
import sys
import os
import math
import random

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("pygame is required: uv add pygame")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1200, 900
FPS = 60
NUM_PARTICLES = 200
NUM_LAYERS = 28

# Colors (matching the shared_style palette)
BG_COLOR = (26, 26, 46)
GRID_COLOR = (55, 55, 79)
TEXT_COLOR = (224, 224, 224)
ACCENT_COLOR = (255, 215, 0)

STYLE_COLORS = {
    "terse":    (33, 150, 243),
    "formal":   (156, 39, 176),
    "socratic": (255, 152, 0),
    "dry-wit":  (76, 175, 80),
}

BASIN_COLORS = {
    "terse":         (33, 150, 243, 30),
    "formal":        (156, 39, 176, 30),
    "socratic":      (255, 152, 0, 30),
    "dry-wit":       (76, 175, 80, 30),
    "cult_of_jason": (233, 30, 99, 20),
}

DEAD_ZONE_COLOR = (189, 189, 189, 20)
SWEET_SPOT_COLOR = (139, 195, 74, 15)
COLLAPSE_COLOR = (244, 67, 54, 15)

# Basin centers in screen coords
CENTER = (WIDTH // 2, HEIGHT // 2)
BASIN_CENTERS = {
    "terse":         (-200, -150),
    "formal":        (200, -200),
    "socratic":      (250, 130),
    "dry-wit":       (-150, 220),
    "cult_of_jason": (-280, -210),
}

# Style axis directions (unit vectors in screen space)
STYLE_DIRECTIONS = {
    "terse":    (-0.75, -0.66),
    "formal":   (0.66, -0.75),
    "socratic": (0.85, 0.53),
    "dry-wit":  (-0.50, 0.87),
}


class Particle:
    """A token position in the residual stream, projected to 2D."""
    def __init__(self, x, y, color=(150, 150, 180)):
        self.base_x = x
        self.base_y = y
        self.x = x
        self.y = y
        self.color = color
        self.trail = []
        self.size = random.uniform(2, 4)

    def update(self, steer_dx, steer_dy, alpha, layer_frac):
        """Apply steering displacement scaled by alpha and layer propagation."""
        # Displacement grows with layer depth (early layers: small; late: large)
        layer_scale = max(0, (layer_frac - 0.3) / 0.7) if layer_frac > 0.3 else 0
        # Add some noise to simulate different token sensitivities
        noise_scale = 0.3 + random.gauss(0, 0.1)

        dx = steer_dx * alpha * layer_scale * noise_scale * 0.5
        dy = steer_dy * alpha * layer_scale * noise_scale * 0.5

        # Check for collapse: at high alpha, particles scatter randomly
        if alpha > 3.0:
            collapse_noise = (alpha - 3.0) * 15
            dx += random.gauss(0, collapse_noise)
            dy += random.gauss(0, collapse_noise)

        self.trail.append((self.x, self.y))
        if len(self.trail) > 8:
            self.trail.pop(0)

        self.x = self.base_x + dx
        self.y = self.base_y + dy

    def reset(self):
        self.x = self.base_x
        self.y = self.base_y
        self.trail = []


def draw_text(screen, text, pos, color=TEXT_COLOR, size=14, bold=False):
    """Draw text on screen."""
    font = pygame.font.SysFont("monospace", size, bold=bold)
    surface = font.render(text, True, color)
    screen.blit(surface, pos)


def draw_circle_alpha(screen, color_rgba, center, radius):
    """Draw a filled circle with alpha blending."""
    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(s, color_rgba, (radius, radius), radius)
    screen.blit(s, (center[0] - radius, center[1] - radius))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Qwen3-0.6B Steering Vector Explorer")
    clock = pygame.time.Clock()

    # ── State ────────────────────────────────────────────────────────────────
    alpha = 0.0
    steer_angle = math.atan2(STYLE_DIRECTIONS["terse"][1], STYLE_DIRECTIONS["terse"][0])
    steer_magnitude = 150.0  # Visual length
    current_style = "terse"
    current_layer = 15
    layer_anim = False
    layer_anim_speed = 0.5
    layer_anim_t = 0.0
    dragging = False

    # ── Create particles (gaussian cluster around center) ────────────────────
    particles = []
    for _ in range(NUM_PARTICLES):
        x = CENTER[0] + random.gauss(0, 40)
        y = CENTER[1] + random.gauss(0, 40)
        # Slight color variation
        r = min(255, max(0, 150 + random.randint(-30, 30)))
        g = min(255, max(0, 150 + random.randint(-30, 30)))
        b = min(255, max(0, 180 + random.randint(-30, 30)))
        particles.append(Particle(x, y, (r, g, b)))

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    alpha = 0.0
                    for p in particles:
                        p.reset()
                elif event.key == pygame.K_UP:
                    alpha = min(alpha + 0.2, 8.0)
                elif event.key == pygame.K_DOWN:
                    alpha = max(alpha - 0.2, 0.0)
                elif event.key == pygame.K_LEFT:
                    steer_angle -= 0.15
                elif event.key == pygame.K_RIGHT:
                    steer_angle += 0.15
                elif event.key == pygame.K_1:
                    current_style = "terse"
                    steer_angle = math.atan2(STYLE_DIRECTIONS["terse"][1], STYLE_DIRECTIONS["terse"][0])
                elif event.key == pygame.K_2:
                    current_style = "formal"
                    steer_angle = math.atan2(STYLE_DIRECTIONS["formal"][1], STYLE_DIRECTIONS["formal"][0])
                elif event.key == pygame.K_3:
                    current_style = "socratic"
                    steer_angle = math.atan2(STYLE_DIRECTIONS["socratic"][1], STYLE_DIRECTIONS["socratic"][0])
                elif event.key == pygame.K_4:
                    current_style = "dry-wit"
                    steer_angle = math.atan2(STYLE_DIRECTIONS["dry-wit"][1], STYLE_DIRECTIONS["dry-wit"][0])
                elif event.key == pygame.K_l:
                    layer_anim = not layer_anim
                    layer_anim_t = 0.0
                elif event.key == pygame.K_TAB:
                    current_layer = (current_layer + 1) % NUM_LAYERS

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    dx = mx - CENTER[0]
                    dy = my - CENTER[1]
                    steer_angle = math.atan2(dy, dx)
                    dist = math.sqrt(dx * dx + dy * dy)
                    alpha = min(dist / 80.0, 8.0)

            # Mouse wheel for alpha
            elif event.type == pygame.MOUSEWHEEL:
                alpha = max(0, min(8.0, alpha + event.y * 0.3))

        # ── Layer animation ──────────────────────────────────────────────────
        if layer_anim:
            layer_anim_t += dt * layer_anim_speed
            if layer_anim_t > 1.0:
                layer_anim_t = 0.0
            current_layer = int(layer_anim_t * (NUM_LAYERS - 1))
        layer_frac = current_layer / (NUM_LAYERS - 1)

        # ── Compute steering vector in screen space ──────────────────────────
        steer_dx = math.cos(steer_angle) * steer_magnitude
        steer_dy = math.sin(steer_angle) * steer_magnitude

        # ── Update particles ─────────────────────────────────────────────────
        for p in particles:
            p.update(steer_dx, steer_dy, alpha, layer_frac)

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Grid
        for x in range(0, WIDTH, 50):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)

        # ── Alpha threshold rings ────────────────────────────────────────────
        # Dead zone ring (alpha < 0.3 equivalent)
        draw_circle_alpha(screen, DEAD_ZONE_COLOR + (20,) if len(DEAD_ZONE_COLOR) == 3 else DEAD_ZONE_COLOR,
                          CENTER, 24)
        # Sweet spot ring
        draw_circle_alpha(screen, SWEET_SPOT_COLOR + (15,) if len(SWEET_SPOT_COLOR) == 3 else SWEET_SPOT_COLOR,
                          CENTER, 200)
        # Collapse ring
        draw_circle_alpha(screen, COLLAPSE_COLOR + (10,) if len(COLLAPSE_COLOR) == 3 else COLLAPSE_COLOR,
                          CENTER, 350)

        # ── Style basin regions ──────────────────────────────────────────────
        for name, (bx, by) in BASIN_CENTERS.items():
            cx, cy = CENTER[0] + bx, CENTER[1] + by
            if name in BASIN_COLORS:
                draw_circle_alpha(screen, BASIN_COLORS[name], (cx, cy), 80)
            label_color = STYLE_COLORS.get(name, (233, 30, 99))
            draw_text(screen, name.replace("_", " "), (cx - 30, cy - 8),
                     color=label_color, size=12, bold=True)

        # ── Draw particle trails ─────────────────────────────────────────────
        for p in particles:
            for i, (tx, ty) in enumerate(p.trail):
                trail_alpha = int(40 * (i + 1) / len(p.trail)) if p.trail else 0
                color = (*p.color, trail_alpha)
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (2, 2), 2)
                screen.blit(s, (int(tx) - 2, int(ty) - 2))

        # ── Draw particles ───────────────────────────────────────────────────
        for p in particles:
            # Color shift: redder when alpha is high (approaching collapse)
            if alpha > 2.5:
                collapse_t = min(1.0, (alpha - 2.5) / 3.0)
                r = min(255, int(p.color[0] * (1 - collapse_t) + 244 * collapse_t))
                g = min(255, int(p.color[1] * (1 - collapse_t) + 67 * collapse_t))
                b = min(255, int(p.color[2] * (1 - collapse_t) + 54 * collapse_t))
                draw_color = (r, g, b)
            else:
                style_color = STYLE_COLORS.get(current_style, (150, 150, 180))
                blend_t = min(1.0, alpha / 2.0)
                r = int(p.color[0] * (1 - blend_t) + style_color[0] * blend_t)
                g = int(p.color[1] * (1 - blend_t) + style_color[1] * blend_t)
                b = int(p.color[2] * (1 - blend_t) + style_color[2] * blend_t)
                draw_color = (min(255, r), min(255, g), min(255, b))

            px, py = int(p.x), int(p.y)
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                pygame.draw.circle(screen, draw_color, (px, py), int(p.size))

        # ── Draw steering vector arrow ───────────────────────────────────────
        if alpha > 0.01:
            arrow_len = alpha * 40
            end_x = CENTER[0] + math.cos(steer_angle) * arrow_len
            end_y = CENTER[1] + math.sin(steer_angle) * arrow_len

            # Arrow shaft
            arrow_color = STYLE_COLORS.get(current_style, ACCENT_COLOR)
            pygame.draw.line(screen, arrow_color, CENTER,
                           (int(end_x), int(end_y)), 3)

            # Arrowhead
            head_angle = 0.4
            head_len = 15
            for sign in [-1, 1]:
                hx = end_x - head_len * math.cos(steer_angle + sign * head_angle)
                hy = end_y - head_len * math.sin(steer_angle + sign * head_angle)
                pygame.draw.line(screen, arrow_color,
                               (int(end_x), int(end_y)),
                               (int(hx), int(hy)), 3)

        # ── Draw center marker ───────────────────────────────────────────────
        pygame.draw.circle(screen, ACCENT_COLOR, CENTER, 5)
        pygame.draw.circle(screen, ACCENT_COLOR, CENTER, 5, 1)

        # ── HUD: status display ──────────────────────────────────────────────
        hud_y = 10
        hud_x = 10
        draw_text(screen, "QWEN3-0.6B STEERING VECTOR EXPLORER", (hud_x, hud_y),
                 color=ACCENT_COLOR, size=16, bold=True)
        hud_y += 25

        draw_text(screen, f"Style:  {current_style}", (hud_x, hud_y),
                 color=STYLE_COLORS.get(current_style, TEXT_COLOR), size=14, bold=True)
        hud_y += 20

        # Alpha with color coding
        if alpha < 0.3:
            alpha_color = (189, 189, 189)
            alpha_label = "DEAD ZONE"
        elif alpha <= 2.5:
            alpha_color = (139, 195, 74)
            alpha_label = "EFFECTIVE"
        elif alpha <= 3.0:
            alpha_color = (255, 152, 0)
            alpha_label = "DANGER"
        else:
            alpha_color = (244, 67, 54)
            alpha_label = "COLLAPSE"

        draw_text(screen, f"Alpha:  {alpha:.2f}  [{alpha_label}]", (hud_x, hud_y),
                 color=alpha_color, size=14, bold=True)
        hud_y += 20

        draw_text(screen, f"Layer:  {current_layer}/{NUM_LAYERS-1}  "
                         f"{'[ANIMATING]' if layer_anim else ''}",
                 (hud_x, hud_y), size=14)
        hud_y += 20

        # SNR estimation
        snr_est = alpha * 19.6 / 488.0 * 100  # Using layer 15 values
        draw_text(screen, f"Est. SNR: {snr_est:.1f}%", (hud_x, hud_y), size=12)
        hud_y += 20

        eff_mag = alpha * 19.6
        draw_text(screen, f"Eff. magnitude: {eff_mag:.1f}", (hud_x, hud_y), size=12)
        hud_y += 20

        res_norm = 488
        draw_text(screen, f"Residual norm: {res_norm}", (hud_x, hud_y), size=12)

        # ── Layer progress bar ───────────────────────────────────────────────
        bar_x, bar_y = WIDTH - 50, 60
        bar_h = HEIGHT - 120
        # Background
        pygame.draw.rect(screen, GRID_COLOR, (bar_x, bar_y, 30, bar_h), 1)
        # Fill to current layer
        fill_h = int(bar_h * (current_layer / (NUM_LAYERS - 1)))
        if fill_h > 0:
            # Gradient fill
            for dy in range(fill_h):
                t = dy / max(bar_h, 1)
                r = int(21 * (1 - t) + 198 * t)
                g = int(101 * (1 - t) + 40 * t)
                b = int(192 * (1 - t) + 40 * t)
                pygame.draw.line(screen, (r, g, b),
                               (bar_x + 1, bar_y + dy),
                               (bar_x + 29, bar_y + dy))
        # Layer markers
        for l in range(NUM_LAYERS):
            ly = bar_y + int(bar_h * l / (NUM_LAYERS - 1))
            marker_color = ACCENT_COLOR if l == current_layer else GRID_COLOR
            pygame.draw.line(screen, marker_color, (bar_x - 2, ly), (bar_x + 32, ly), 1)
            if l % 4 == 0 or l == current_layer:
                draw_text(screen, str(l), (bar_x - 20, ly - 6), size=10,
                         color=ACCENT_COLOR if l == current_layer else TEXT_COLOR)

        draw_text(screen, "Layer", (bar_x - 5, bar_y - 18), size=10, bold=True)

        # Sweet spot annotation on layer bar
        ss_y1 = bar_y + int(bar_h * 12 / (NUM_LAYERS - 1))
        ss_y2 = bar_y + int(bar_h * 18 / (NUM_LAYERS - 1))
        ss_surface = pygame.Surface((30, ss_y2 - ss_y1), pygame.SRCALPHA)
        ss_surface.fill((139, 195, 74, 30))
        screen.blit(ss_surface, (bar_x, ss_y1))

        # ── Controls help ────────────────────────────────────────────────────
        help_y = HEIGHT - 140
        help_x = 10
        controls = [
            "Controls:",
            "  Up/Down: alpha +/- 0.2",
            "  Left/Right: rotate vector",
            "  1-4: style axis",
            "  Space: reset",
            "  L: toggle layer animation",
            "  Tab: next layer",
            "  Mouse drag: direct control",
            "  Scroll: alpha fine-tune",
            "  Esc: quit",
        ]
        for i, line in enumerate(controls):
            draw_text(screen, line, (help_x, help_y + i * 14), size=10,
                     color=TEXT_COLOR if i > 0 else ACCENT_COLOR)

        # ── Phase indicator ──────────────────────────────────────────────────
        phase_x = WIDTH - 250
        phase_y = HEIGHT - 60
        phases = [
            ("DEAD", alpha < 0.3, (189, 189, 189)),
            ("EFFECTIVE", 0.3 <= alpha <= 2.5, (139, 195, 74)),
            ("COLLAPSE", alpha > 3.0, (244, 67, 54)),
        ]
        draw_text(screen, "Phase:", (phase_x, phase_y), size=12, bold=True)
        px = phase_x + 60
        for label, active, color in phases:
            text_color = color if active else (80, 80, 80)
            draw_text(screen, label, (px, phase_y), size=12, bold=active,
                     color=text_color)
            px += 80

        pygame.display.flip()

    pygame.quit()
    print("Pygame window closed.")


if __name__ == "__main__":
    main()
