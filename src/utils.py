from typing import NamedTuple
from EnvReceiver import EnvReceiver
from src.wrappers import *

class AgentConfig(NamedTuple):
    state_dims: int
    action_c_dims: int
    action_d_dims: int
    action_scale: int
    action_bias: int
    frame_stack: int
    lr: float
    entropy_lr: float
    gamma: float          # reward decay
    tau: float            # target network update rate
    memory_size: int
    batch_size: int
    device: str

    def __repr__(self):
        return f"AgentConfig(frame_stack={self.frame_stack}, lr={self.lr}, gamma={self.gamma}, tau={self.tau}, memory_size={self.memory_size}, batch_size={self.batch_size})"
    
def make_env(frame_stack):
    env = EnvReceiver()
    env = PIL2TorchWrapper(env)
    env = FrameStackWrapper(env, frame_stack)
    obs_shape = env.reset().shape
    return env, obs_shape