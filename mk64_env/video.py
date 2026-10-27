"""Write emulator frames to an MP4 file."""

from __future__ import annotations

import os

import imageio.v2 as imageio
import numpy as np


def stretch_frame(frame: np.ndarray) -> np.ndarray:
    """Double the rows of a 640x240 field so the image has the game's 4:3 aspect.

    Frames that are already at least half as tall as they are wide are
    returned unchanged.
    """
    if frame.shape[1] >= 2 * frame.shape[0]:
        return np.repeat(frame, 2, axis=0)
    return frame


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
        self._writer.append_data(stretch_frame(frame))

    def close(self) -> None:
        """Finish the file."""
        self._writer.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
