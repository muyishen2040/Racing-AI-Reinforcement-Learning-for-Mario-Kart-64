"""Write emulator frames to an MP4 file."""

from __future__ import annotations

import os

import imageio.v2 as imageio
import numpy as np


class VideoRecorder:
    """Accumulate RGB frames and write them as H.264 MP4 on close().

    Angrylion outputs 640x240 fields; frames are stretched to 640x480 so the
    video has the game's 4:3 aspect ratio.
    """

    def __init__(self, path: str | os.PathLike[str], fps: int = 60):
        """Open `path` for writing at `fps` frames per second."""
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def add(self, frame: np.ndarray) -> None:
        """Append one (H, W, 3) uint8 frame."""
        if frame.shape[1] >= 2 * frame.shape[0]:
            frame = np.repeat(frame, 2, axis=0)
        self._writer.append_data(frame)

    def close(self) -> None:
        """Finish the file."""
        self._writer.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
