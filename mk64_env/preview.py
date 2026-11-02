"""Optional live display of the full-colour emulator frames."""

from __future__ import annotations

import numpy as np

from mk64_env.video import stretch_frame


class LivePreview:
    """Display occasional frames without changing the agent's observations."""

    def __init__(self, every: int = 4):
        if every < 1:
            raise ValueError("preview interval must be at least 1")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Live preview needs pygame: pip install -e ".[preview]"') from exc
        self.pygame = pygame
        self.every = every
        self.enabled = True
        pygame.display.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Mario Kart 64 live preview")

    def show(self, frame: np.ndarray, step: int) -> None:
        if not self.enabled:
            return
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                return
        if step % self.every:
            return
        image = self.pygame.surfarray.make_surface(np.transpose(stretch_frame(frame), (1, 0, 2)))
        self.screen.blit(self.pygame.transform.scale(image, (640, 480)), (0, 0))
        self.pygame.display.flip()

    def close(self) -> None:
        if self.enabled:
            self.enabled = False
            self.pygame.display.quit()
