import torch
import torch.nn as nn


class DA_Loss(nn.Module):
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

        # Mean difference (1st moment)
        loss = torch.norm(h_full.mean(dim=0) - h_miss.mean(dim=0), p=2)

        # Higher-order central moments
        for k in range(2, self.n_moments + 1):
            loss = loss + self._moment_diff(h_full, h_miss, k)

        return loss
