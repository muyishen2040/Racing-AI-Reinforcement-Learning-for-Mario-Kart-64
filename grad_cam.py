"""Render a Grad-CAM video of what the actor attends to in a recorded episode.

Usage: python grad_cam.py demo.pkl actor.pth [--output episode_gradcam.mp4]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from mk64_env.video import VideoRecorder
from src.model import ActorNet


class ActorNetVisualize(ActorNet):
    """ActorNet that keeps the last conv activations and their gradients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.activations = None

    def forward(self, state):
        """Same as ActorNet.forward, with a hook on the conv output."""
        x = self.conv_layers(state)
        x.register_hook(lambda grad: setattr(self, "gradients", grad))
        self.activations = x
        x = F.relu(self.fc(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        binary_logits = self.binary_logits(x)
        binary_logits[:, 0] += self.forward_bias
        return mean, log_std.exp(), binary_logits


def grad_cam(net: ActorNetVisualize, state: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Return a [0, 1] heat map of `size` (height, width) for the mean action."""
    net.zero_grad()
    mean, _, _ = net(state)
    mean.mean().backward()
    gradients = net.gradients.view(1, 128, 12, 12)
    activations = net.activations.view(1, 128, 12, 12)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False).squeeze()
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def overlay(frame: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a heat map over an RGB frame."""
    heat = ImageOps.colorize(Image.fromarray(np.uint8(255 * cam)), black="blue", white="red")
    gray = ImageOps.colorize(Image.fromarray(frame).convert("L"), black="black", white="white")
    return np.asarray(Image.blend(gray, heat, alpha=0.5))


def main() -> None:
    """Write the Grad-CAM video for every step of the recording."""
    import pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("actor")
    parser.add_argument("--output", default="episode_gradcam.mp4")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorNetVisualize(80).to(device)
    net.load_state_dict(torch.load(args.actor, map_location=device))
    net.eval()
    with open(args.recording, "rb") as f:
        data = pickle.load(f)
    with VideoRecorder(args.output, fps=15) as recorder:
        for obs, frame in tqdm(zip(data["obs"], data["frames"]), total=len(data["obs"]), desc="Grad-CAM"):
            state = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255).to(device)
            cam = grad_cam(net, state, frame.shape[:2])
            recorder.add(overlay(frame, cam))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
