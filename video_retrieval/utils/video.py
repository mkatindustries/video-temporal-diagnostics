"""Video loading and frame extraction utilities."""

from pathlib import Path

import av
import numpy as np
import torch


def load_video(
    video_path: str | Path,
    max_frames: int | None = None,
    sample_rate: int = 1,
    max_resolution: int | None = 720,
) -> tuple[list[np.ndarray], float]:
    """Load video frames with uniform sampling across full duration.

    When max_frames is set, frames are sampled uniformly across the entire
    video rather than taking the first max_frames. This ensures the full
    temporal extent of the video is represented.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract (None = all after
            sample_rate). Frames are uniformly spaced across the video.
        sample_rate: Sample every Nth frame (applied before max_frames).
        max_resolution: Maximum height (preserves aspect ratio). None = original.

    Returns:
        Tuple of (list of RGB frames as numpy arrays, fps).
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30)

    # First pass: count total frames to enable uniform sampling
    total_frames = stream.frames
    if total_frames == 0:
        # Some containers don't report frame count; must count manually
        total_frames = 0
        for _ in container.decode(video=0):
            total_frames += 1
        container.close()
        container = av.open(str(video_path))
        stream = container.streams.video[0]

    # Determine which original frame indices to keep
    # Step 1: apply sample_rate to get candidate indices
    candidate_indices = list(range(0, total_frames, sample_rate))

    # Step 2: if max_frames is set and we have more candidates, subsample uniformly
    if max_frames and len(candidate_indices) > max_frames:
        uniform = np.linspace(0, len(candidate_indices) - 1, max_frames).astype(int)
        candidate_indices = [candidate_indices[i] for i in uniform]

    keep_set = set(candidate_indices)

    frames = []
    frame_idx = 0

    for frame in container.decode(video=0):
        if frame_idx in keep_set:
            img = frame.to_ndarray(format="rgb24")

            # Resize if needed
            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                import cv2
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)

            if len(frames) >= len(candidate_indices):
                break

        frame_idx += 1

    container.close()
    return frames, fps


def extract_frames(
    video_path: str | Path,
    frame_indices: list[int] | None = None,
    sample_rate: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Extract specific frames from a video.

    Args:
        video_path: Path to video file.
        frame_indices: Specific frame indices to extract (None = use sample_rate).
        sample_rate: Sample every Nth frame (ignored if frame_indices provided).
        max_frames: Maximum frames to return.

    Returns:
        List of RGB frames as numpy arrays.
    """
    container = av.open(str(video_path))

    if frame_indices:
        frame_set = set(frame_indices)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_set:
                frames.append(frame.to_ndarray(format="rgb24"))
                if max_frames and len(frames) >= max_frames:
                    break
    else:
        frames, _ = load_video(video_path, max_frames=max_frames, sample_rate=sample_rate)

    container.close()
    return frames


def frames_to_tensor(
    frames: list[np.ndarray],
    normalize: bool = True,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Convert frames to a batched tensor.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        normalize: Apply ImageNet normalization.
        device: Target device.

    Returns:
        Tensor of shape (N, 3, H, W).
    """
    tensors = []
    for frame in frames:
        # HWC -> CHW, normalize to [0, 1]
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors).to(device)

    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        batch = (batch - mean) / std

    return batch
