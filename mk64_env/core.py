"""Headless Mupen64Plus-Next emulator core driven through libretro.py.

The core renders with the Angrylion software RDP, so no display server or GPU
is needed. Frames, controller input, RDRAM, and savestates are all exchanged
in-process through the libretro API.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from libretro import (
    AnalogState,
    ArrayVideoDriver,
    DictOptionDriver,
    FrameBufferSpecial,
    IterableInputDriver,
    JoypadState,
    Session,
)
from libretro.api.memory import RETRO_MEMORY_SYSTEM_RAM
from libretro.drivers.input.iterable import PortState

# Core options that select the headless software renderer. The 43screensize
# option only affects GLideN64; Angrylion always outputs one 640x240 field.
CORE_OPTIONS = {
    "mupen64plus-rdp-plugin": "angrylion",
    "mupen64plus-rsp-plugin": "parallel",
    "mupen64plus-cpucore": "dynamic_recompiler",
    "mupen64plus-angrylion-multithread": "4",
    "mupen64plus-astick-deadzone": "0",
}

JOYSTICK_MAX = 80
"""Largest joystick magnitude the N64 game reads, in N64 units."""

_ANALOG_SCALE = 32767 / JOYSTICK_MAX


@dataclass(frozen=True)
class ControllerInput:
    """One frame of N64 controller state.

    joy_x and joy_y use N64 units in [-80, 80]. Positive joy_y points up,
    matching the N64 convention. All buttons are booleans.
    """

    joy_x: int = 0
    joy_y: int = 0
    a: bool = False
    b: bool = False
    z: bool = False
    r: bool = False
    l: bool = False
    start: bool = False
    c_up: bool = False
    c_down: bool = False
    c_left: bool = False
    c_right: bool = False

    def to_port_state(self) -> PortState:
        """Map this input onto the core's RetroPad layout.

        Mupen64Plus-Next maps N64 A to RetroPad B, N64 B to RetroPad Y,
        Z to L2, and the C buttons to the right analog stick.
        """
        x = int(max(-JOYSTICK_MAX, min(JOYSTICK_MAX, self.joy_x)) * _ANALOG_SCALE)
        y = int(max(-JOYSTICK_MAX, min(JOYSTICK_MAX, self.joy_y)) * _ANALOG_SCALE)
        cx = (32767 if self.c_right else 0) - (32767 if self.c_left else 0)
        cy = (32767 if self.c_down else 0) - (32767 if self.c_up else 0)
        return PortState(
            joypad=JoypadState(b=self.a, y=self.b, l2=self.z, r=self.r, l=self.l, start=self.start),
            analog=AnalogState(lstick=(x, -y), rstick=(cx, cy)),
        )


NO_INPUT = ControllerInput()


class _FrameVideoDriver(ArrayVideoDriver):
    """ArrayVideoDriver with a zero-copy numpy view of the last frame."""

    def rgb_frame(self) -> np.ndarray:
        """Return the last rendered frame as an (H, W, 3) uint8 RGB array.

        The core outputs XRGB8888, which is B, G, R, X in memory on a
        little-endian host. The returned array is a copy.
        """
        if self._frame is None or self._frame_dims is None:
            raise RuntimeError("no frame has been rendered yet")
        dims = self._frame_dims
        raw = np.frombuffer(self._frame, dtype=np.uint8)[: dims.height * dims.pitch]
        pixels = raw.reshape(dims.height, dims.pitch)[:, : dims.width * 4].reshape(dims.height, dims.width, 4)
        return np.ascontiguousarray(pixels[:, :, 2::-1])


class _LenientOptionDriver(DictOptionDriver):
    """DictOptionDriver that tolerates options without a default value.

    The nightly core registers ParaLLEl-RDP options with no default when
    Vulkan is compiled out, which trips an assertion in libretro.py 0.12.
    """

    def set_display(self, key: bytes, visible: bool) -> None:
        definition = self._options_us.get(key)
        if definition is not None and key not in self._options and definition.default_value is None:
            return
        super().set_display(key, visible)


class _Session(Session):
    """Session that treats a NULL frame pointer as a duplicated frame.

    libretro.py 0.12 compares the pointer value against 0, but ctypes reports
    NULL as None, so the Angrylion frame-dupe path raised a TypeError.
    """

    def video_refresh(self, data, width: int, height: int, pitch: int) -> None:
        if data.value is None:
            self._video.refresh(FrameBufferSpecial.DUPE, width, height, pitch)
            return
        super().video_refresh(data, width, height, pitch)


class EmulatorCore:
    """A running Mupen64Plus-Next instance with frame-step control.

    Each instance loads a private copy of the core shared library, so several
    instances can coexist in one process. Call start() before use and close()
    when done, or use the instance as a context manager.
    """

    def __init__(self, core_path: str | os.PathLike[str], rom_path: str | os.PathLike[str]):
        """Record the core and ROM paths without loading anything.

        Inputs: path to mupen64plus_next_libretro.so and path to the ROM.
        """
        self.core_path = Path(core_path)
        self.rom_path = Path(rom_path)
        if not self.core_path.is_file():
            raise FileNotFoundError(f"libretro core not found: {self.core_path}")
        if not self.rom_path.is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom_path}")
        self._input = NO_INPUT
        self._video = _FrameVideoDriver()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._session: Session | None = None
        self._ram: memoryview | None = None
        self.frame_count = 0

    def _input_source(self):
        while True:
            yield [self._input.to_port_state()]

    def start(self) -> None:
        """Load a private copy of the core, load the ROM, and run one frame."""
        if self._session is not None:
            return
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mk64core_")
        private_core = Path(self._tmpdir.name) / self.core_path.name
        shutil.copy2(self.core_path, private_core)
        session = _Session(
            core=str(private_core),
            game=str(self.rom_path),
            video=self._video,
            input=IterableInputDriver(self._input_source),
            options=_LenientOptionDriver(variables=CORE_OPTIONS),
            perf=None,
        )
        self._session = session.__enter__()
        self.run_frame()
        self._ram = self._session.core.get_memory(RETRO_MEMORY_SYSTEM_RAM)
        if self._ram is None:
            raise RuntimeError("core did not expose RDRAM")

    def close(self) -> None:
        """Unload the core and delete the private copy of the shared library."""
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._ram = None

    def __enter__(self) -> EmulatorCore:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def session(self) -> Session:
        """The live libretro.py session. Raises if start() was not called."""
        if self._session is None:
            raise RuntimeError("EmulatorCore is not started")
        return self._session

    def set_input(self, controller: ControllerInput) -> None:
        """Set the controller state the core reads on every following frame."""
        self._input = controller

    def run_frame(self) -> None:
        """Advance emulation by one video frame (1/60 s)."""
        self.session.run()
        self.frame_count += 1

    def run_frames(self, count: int, controller: ControllerInput | None = None) -> None:
        """Advance `count` frames, optionally holding `controller` throughout."""
        if controller is not None:
            self.set_input(controller)
        for _ in range(count):
            self.run_frame()

    def frame(self) -> np.ndarray:
        """Return the last rendered frame as an (H, W, 3) uint8 RGB array."""
        return self._video.rgb_frame()

    def save_state(self) -> bytes:
        """Serialize the full emulator state."""
        size = self.session.core.serialize_size()
        buffer = bytearray(size)
        if not self.session.core.serialize(buffer):
            raise RuntimeError("retro_serialize failed")
        return bytes(buffer)

    def load_state(self, state: bytes) -> None:
        """Restore a state produced by save_state()."""
        if not self.session.core.unserialize(state):
            raise RuntimeError("retro_unserialize failed")

    # RDRAM accessors. The core stores RDRAM as host-endian 32-bit words, so a
    # byte at N64 address A lives at host offset (A & 0x7FFFFF) ^ 3.
    def _ram_bytes(self) -> memoryview:
        if self._ram is None:
            raise RuntimeError("EmulatorCore is not started")
        return self._ram

    def read_u8(self, address: int) -> int:
        """Read an unsigned byte at an N64 virtual address (0x80xxxxxx)."""
        return self._ram_bytes()[(address & 0x7FFFFF) ^ 3]

    def read_u16(self, address: int) -> int:
        """Read a big-endian unsigned 16-bit value."""
        return (self.read_u8(address) << 8) | self.read_u8(address + 1)

    def read_s16(self, address: int) -> int:
        """Read a big-endian signed 16-bit value."""
        value = self.read_u16(address)
        return value - 0x10000 if value & 0x8000 else value

    def read_s32(self, address: int) -> int:
        """Read a 4-byte-aligned signed 32-bit value."""
        return struct.unpack_from("<i", self._ram_bytes(), address & 0x7FFFFF)[0]

    def read_u32(self, address: int) -> int:
        """Read a 4-byte-aligned unsigned 32-bit value."""
        return struct.unpack_from("<I", self._ram_bytes(), address & 0x7FFFFF)[0]

    def read_f32(self, address: int) -> float:
        """Read a 4-byte-aligned IEEE float."""
        return struct.unpack_from("<f", self._ram_bytes(), address & 0x7FFFFF)[0]
