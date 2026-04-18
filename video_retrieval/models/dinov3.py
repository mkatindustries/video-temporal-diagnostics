"""DINOv3 encoder for video frame embeddings.

Uses HuggingFace transformers for automatic weight downloading.
Provides methods for extracting:
- Global CLS token embeddings (semantic)
- Patch token features (spatial/structural)
- Attention maps (where the model looks)
- Attention centroids (for trajectory analysis)
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DINOv3ViTImageProcessor, DINOv3ViTModel


class DINOv3Encoder:
    """DINOv3 encoder for video frame embeddings.

    Uses HuggingFace transformers - weights are automatically downloaded.
    Provides methods for extracting embeddings and attention-based trajectories.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str | torch.device = "cuda",
    ):
        """Initialize DINOv3 encoder.

        Args:
            model_name: HuggingFace model name. Options:
                - "facebook/dinov3-vits16-pretrain-lvd1689m" (ViT-S, 22M params)
                - "facebook/dinov3-vitb16-pretrain-lvd1689m" (ViT-B, 86M params)
                - "facebook/dinov3-vitl16-pretrain-lvd1689m" (ViT-L, 300M params, 1024-dim) [default]
                - "facebook/dinov3-vith16-pretrain-lvd1689m" (ViT-H+, ~600M params)
            device: Target device.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name

        # Load model and processor from HuggingFace (auto-downloads weights)
        # Note: output_attentions=True must be set at load time for attention extraction
        self.processor: Any = DINOv3ViTImageProcessor.from_pretrained(model_name)
        self.model = DINOv3ViTModel.from_pretrained(
            model_name, output_attentions=True
        ).eval().to(self.device)

        # Get config
        self.embedding_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_heads = self.model.config.num_attention_heads
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)

    def _preprocess(self, frames: list[np.ndarray]) -> dict[str, torch.Tensor]:
        """Preprocess frames for the model.

        Args:
            frames: List of RGB frames (H, W, 3) as numpy arrays.

        Returns:
            Dict with 'pixel_values' tensor ready for model.
        """
        # HuggingFace processor handles resizing, normalization, etc.
        inputs = self.processor(images=frames, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode_frames(
        self,
        frames: list[np.ndarray],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode frames to global (CLS) embeddings.

        Args:
            frames: List of RGB frames (H, W, 3).
            batch_size: Batch size for processing.
            normalize: L2-normalize embeddings.

        Returns:
            Embeddings tensor (N, embedding_dim).
        """
        all_embeddings = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inputs = self._preprocess(batch_frames)

            outputs = self.model(**inputs)
            # CLS token is at position 0
            embeddings = outputs.last_hidden_state[:, 0]

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_video(
        self,
        frames: list[np.ndarray],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode video frames to embeddings.

        Args:
            frames: List of RGB frames.
            batch_size: Batch size for processing.

        Returns:
            Embeddings tensor (N, embedding_dim).
        """
        return self.encode_frames(frames, batch_size=batch_size)

    @torch.no_grad()
    def get_attention_centroids(
        self,
        frames: list[np.ndarray],
        layer: int = -1,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Compute attention centroid trajectory.

        For each frame, computes the spatial center-of-mass of the CLS token's
        attention over patch tokens. This trajectory encodes camera/subject motion.

        Args:
            frames: List of RGB frames.
            layer: Attention layer to use (-1 = last).
            batch_size: Batch size for processing.

        Returns:
            Centroid positions (N, 2) where columns are (x, y) in [0, 1].
        """
        all_centroids = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inputs = self._preprocess(batch_frames)

            # Get attention weights (enabled via output_attentions=True at load time)
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of (B, heads, N, N) per layer

            # Select layer
            attn = attentions[layer]  # (B, heads, N, N)

            # Average over heads
            attn = attn.mean(dim=1)  # (B, N, N)

            # Get CLS attention over patches (CLS is at position 0)
            # Skip CLS token (position 0) and register tokens (positions 1 to num_register_tokens)
            skip_tokens = 1 + self.num_register_tokens  # CLS + register tokens
            cls_attn = attn[:, 0, skip_tokens:]  # (B, num_patches)

            # Compute spatial dimensions from number of patches
            num_patches = cls_attn.shape[1]
            h = w = int(num_patches**0.5)

            # Reshape to spatial grid
            cls_attn = cls_attn.view(-1, h, w)  # (B, h, w)

            # Normalize to probability distribution
            cls_attn = cls_attn / (cls_attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Compute centroids
            y_coords = torch.arange(h, device=self.device).float()
            x_coords = torch.arange(w, device=self.device).float()

            centroid_y = (cls_attn.sum(dim=2) * y_coords).sum(dim=1) / h
            centroid_x = (cls_attn.sum(dim=1) * x_coords).sum(dim=1) / w

            centroids = torch.stack([centroid_x, centroid_y], dim=1)  # (B, 2)
            all_centroids.append(centroids)

        return torch.cat(all_centroids, dim=0)

    @torch.no_grad()
    def get_patch_features(
        self,
        frames: list[np.ndarray],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Get patch token features for each frame.

        Args:
            frames: List of RGB frames.
            batch_size: Batch size for processing.

        Returns:
            Patch features (N, num_patches, embedding_dim).
        """
        all_patches = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inputs = self._preprocess(batch_frames)

            outputs = self.model(**inputs)
            # Skip CLS token at position 0
            patch_features = outputs.last_hidden_state[:, 1:]
            all_patches.append(patch_features)

        return torch.cat(all_patches, dim=0)

    @torch.no_grad()
    def get_patch_statistics(
        self,
        frames: list[np.ndarray],
        batch_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Compute patch-level statistics for each frame.

        Returns variance and entropy of patch tokens - useful for
        capturing spatial texture independent of semantics.

        Args:
            frames: List of RGB frames.
            batch_size: Batch size for processing.

        Returns:
            Dict with 'variance', 'entropy' tensors of shape (N,).
        """
        all_variance = []
        all_entropy = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inputs = self._preprocess(batch_frames)

            outputs = self.model(**inputs)
            patch_features = outputs.last_hidden_state[:, 1:]  # (B, num_patches, D)

            # Variance across patches (spatial complexity)
            variance = patch_features.var(dim=1).mean(dim=1)  # (B,)
            all_variance.append(variance)

            # Entropy of patch token distribution
            # Use softmax over patches as pseudo-probability
            probs = F.softmax(patch_features.norm(dim=2), dim=1)  # (B, num_patches)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (B,)
            all_entropy.append(entropy)

        return {
            "variance": torch.cat(all_variance, dim=0),
            "entropy": torch.cat(all_entropy, dim=0),
        }
