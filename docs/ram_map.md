# Mario Kart 64 (US) RAM map used by mk64_env

All addresses are N64 virtual addresses. `mk64_env/ram.py` holds the constants.
The core exposes RDRAM as host-endian 32-bit words, so `EmulatorCore.read_*`
XOR the byte offset with 3.

| Symbol | Address | Type | Source | Check |
|---|---|---|---|---|
| gPlayers[0] | 0x800F6990 | Player struct, stride 0xDD8 | RAM scan for consecutive pointers P, P+0xDD8, ... at 0x800DC4DC (gPlayerOne..gPlayerEight) | pos and currentSpeed change while driving |
| gRaceState | 0x800DC512 | u16 | RAM diff: the only halfword that goes 2 (staging) to 3 (in progress) at the green light and stays 3 | decomp defines RACE_STAGING=2, RACE_IN_PROGRESS=3 |
| gCourseTimer | 0x800DC598 | f32 seconds | hack64 wiki | starts counting at the green light |
| gCurrentCourseId | 0x800DC5A0 | s16 | hack64 wiki | reads 8 (COURSE_LUIGI_RACEWAY) on Luigi Raceway |
| gLapCountByPlayerId | 0x80164390 | s32[10] | decomp comment `D_80164390` in src/cpu_vehicles_camera_path.c | -1 at the start, 0 after crossing the line |
| gGPCurrentRaceRankByPlayerId | 0x801643B8 | s32[10] | decomp comment `D_801643B8` | 0 in Time Trials |
| gCourseCompletionPercentByPlayerId | 0x801644D0 | f32[10] | decomp comment `D_801644D0` | rises from -0.003 to 0.28 in the first 15 s of driving, then stalls at a wall |

Player struct offsets (decomp include/common_structs.h): type 0x00 (u16),
currentRank 0x04 (s16), lapCount 0x08 (s16), pos 0x14 (Vec3f),
rotation 0x2C (Vec3s, yaw at 0x2E), speed 0x94 (f32), currentSpeed 0x9C (f32,
0 to 310), topSpeed 0x214 (f32).

The progress value is lap count plus the fraction of the current lap, so its
per-step difference is the reward signal. Sources: the n64decomp/mk64
repository and https://hack64.net/wiki/doku.php?id=mario_kart_64:memory_map.
