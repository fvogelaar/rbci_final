"""
Visual feedback primitives
"""

import pygame


class VisualBar:
    def __init__(self, width=500, height=200):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height

    def update(self, value):
        """
        value in [-1, 1]
        """
        pygame.event.pump()  # for pygame.event.get() to process events
        value = max(-1.0, min(1.0, value))
        self.screen.fill((0, 0, 0))
        center = self.width // 2
        bar_width = int(center * abs(value))
        x = center if value >= 0 else center - bar_width
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            (x, 0, bar_width, self.height)
        )
        pygame.draw.line(
            self.screen, (255, 255, 255),
            (center, 0), (center, self.height), 2)
        pygame.display.flip()
