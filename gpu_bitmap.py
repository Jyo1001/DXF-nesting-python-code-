
"""GPU-accelerated bitmap helpers for the DXF nesting engine."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    F = None  # type: ignore


def torch_available() -> bool:
    """Return True when PyTorch can be imported."""
    return torch is not None and F is not None


def cuda_available() -> bool:
    return torch_available() and bool(torch.cuda.is_available())


def detect_best_device(preferred: Optional[str] = None) -> Optional["torch.device"]:
    """Return a CUDA device when available (and optionally respect *preferred*)."""
    if not torch_available():
        return None
    try:
        if preferred is not None:
            device = torch.device(preferred)
            if device.type == "cuda":
                if torch.cuda.is_available():
                    return device
                return None
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        return None
    return None


@dataclass
class TorchMaskOps:
    """Operations on bitmap masks backed by PyTorch tensors."""

    device: "torch.device"

    def __post_init__(self) -> None:
        if not torch_available():  # pragma: no cover - import guard
            raise RuntimeError("PyTorch is not available")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        self._offset_cache: dict[int, "torch.Tensor"] = {}

    def zeros(self, height: int, width: int) -> "torch.Tensor":
        return torch.zeros((height, width), dtype=torch.bool, device=self.device)

    def mask_to_tensor(self, mask) -> "torch.Tensor":
        if isinstance(mask, torch.Tensor):
            return mask.to(self.device, dtype=torch.bool)
        if not mask:
            return self.zeros(0, 0)
        rows = [list(row) for row in mask]
        return torch.tensor(rows, dtype=torch.bool, device=self.device)

    def count_true(self, mask: "torch.Tensor") -> int:
        return int(mask.sum().item())

    def find_first_fit(self, occ: "torch.Tensor", part: "torch.Tensor") -> Optional[Tuple[int, int]]:
        ph, pw = part.shape
        H, W = occ.shape
        if ph == 0 or pw == 0:
            return 0, 0
        if ph > H or pw > W:
            return None
        occ_f = occ.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        part_f = part.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        conv = F.conv2d(occ_f, part_f)
        valid = conv.eq(0)
        if not bool(valid.any().item()):
            return None
        idx = torch.nonzero(valid, as_tuple=False)[0]
        y = int(idx[2].item())
        x = int(idx[3].item())
        return x, y

    def or_mask(self, occ: "torch.Tensor", part: "torch.Tensor", ox: int, oy: int) -> None:
        ph, pw = part.shape
        if ph == 0 or pw == 0:
            return
        occ[oy:oy + ph, ox:ox + pw] |= part

    def _disk_offsets(self, r: int) -> "torch.Tensor":
        if r <= 0:
            return torch.zeros((1, 2), dtype=torch.long, device=self.device)
        cached = self._offset_cache.get(r)
        if cached is not None:
            return cached
        coords = []
        rr = r * r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= rr:
                    coords.append((dy, dx))
        tensor = torch.tensor(coords, dtype=torch.long, device=self.device) if coords else torch.zeros((0, 2), dtype=torch.long, device=self.device)
        self._offset_cache[r] = tensor
        return tensor

    def or_dilated(self, occ: "torch.Tensor", part: "torch.Tensor", ox: int, oy: int, radius: int) -> None:
        coords = torch.nonzero(part, as_tuple=False)
        if coords.numel() == 0:
            return
        offsets = self._disk_offsets(radius)
        pts = coords.unsqueeze(1) + offsets.unsqueeze(0)
        ys = pts[..., 0].reshape(-1) + oy
        xs = pts[..., 1].reshape(-1) + ox
        H, W = occ.shape
        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if valid.any():
            occ[ys[valid], xs[valid]] = True


@lru_cache(maxsize=4)
def build_mask_ops(device_str: Optional[str] = None) -> Optional[TorchMaskOps]:
    if not torch_available():  # pragma: no cover - optional dep guard
        return None
    try:
        device = detect_best_device(device_str)
        if device is None:
            return None
        return TorchMaskOps(device)
    except Exception:
        return None
