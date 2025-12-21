import torch
import torch.nn as nn


class CMD_Loss(nn.Module):
    """
    Central Moment Discrepancy (CMD) for aligning full/missing domain features.
    """

    def __init__(self, n_moments: int = 5):
        super().__init__()
        self.n_moments = n_moments

    def _moment_diff(self, h_full: torch.Tensor, h_miss: torch.Tensor, order: int) -> torch.Tensor:
        full_centered = h_full - h_full.mean(dim=0, keepdim=True)
        miss_centered = h_miss - h_miss.mean(dim=0, keepdim=True)
        full_moment = (full_centered ** order).mean(dim=0)
        miss_moment = (miss_centered ** order).mean(dim=0)
        return torch.norm(full_moment - miss_moment, p=2)

    def forward(self, H_full: torch.Tensor, H_miss: torch.Tensor) -> torch.Tensor:
        # Flatten any temporal dimension so CMD works over (B, D)
        h_full = H_full.reshape(H_full.size(0), -1)
        h_miss = H_miss.reshape(H_miss.size(0), -1)

        # L2 normalize to focus CMD on distribution shape rather than scale.
        h_full = torch.nn.functional.normalize(h_full, p=2, dim=1)
        h_miss = torch.nn.functional.normalize(h_miss, p=2, dim=1)

        # Mean difference (1st moment)
        loss = torch.norm(h_full.mean(dim=0) - h_miss.mean(dim=0), p=2)

        # Higher-order central moments
        for k in range(2, self.n_moments + 1):
            loss = loss + self._moment_diff(h_full, h_miss, k)

        return loss


class SWD_Loss(nn.Module):
    """
    Sliced Wasserstein Discrepancy (SWD) for aligning full/missing domain features.
    """

    def __init__(self, num_projections: int = 128, p: int = 2):
        super().__init__()
        if not isinstance(num_projections, int) or num_projections <= 0:
            raise ValueError(f"num_projections must be a positive integer, got {num_projections}")
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, got {p}")
        self.num_projections = num_projections
        self.p = p

    def forward(self, H_full: torch.Tensor, H_miss: torch.Tensor) -> torch.Tensor:
        # Flatten to (B, D)
        h_full = H_full.reshape(H_full.size(0), -1)
        h_miss = H_miss.reshape(H_miss.size(0), -1)

        # Check batch size consistency
        if h_full.size(0) != h_miss.size(0):
            raise ValueError(f"Batch sizes must match for SWD: {h_full.size(0)} vs {h_miss.size(0)}")

        # L2 normalize features
        h_full = torch.nn.functional.normalize(h_full, p=2, dim=1)
        h_miss = torch.nn.functional.normalize(h_miss, p=2, dim=1)

        B, D = h_full.size()

        # Random projections (D, num_projections)
        theta = torch.randn(D, self.num_projections, device=h_full.device, dtype=h_full.dtype)
        theta = torch.nn.functional.normalize(theta, p=2, dim=0)

        proj_full = h_full @ theta  # (B, P)
        proj_miss = h_miss @ theta  # (B, P)

        proj_full_sorted, _ = torch.sort(proj_full, dim=0)
        proj_miss_sorted, _ = torch.sort(proj_miss, dim=0)

        diff = torch.abs(proj_full_sorted - proj_miss_sorted)
        dist = diff if self.p == 1 else diff ** self.p

        loss = dist.mean()
        return loss
