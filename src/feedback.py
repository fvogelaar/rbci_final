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
        value in [0, 1]
        """
        pygame.event.pump()  #or pygame.event.get() to process events
        value = max(0.0, min(1.0, value))
        self.screen.fill((0, 0, 0))
        bar_width = int(self.width * value)
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            (0, 0, bar_width, self.height)
        )
        pygame.display.flip()
