import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "FocalLoss",
]


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        ignore_index=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        targets = targets.long()
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            logits = logits[mask]
            targets = targets[mask]
            if targets.numel() == 0:
                return logits.sum()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1.0 - pt).pow(self.gamma)
        ce_loss = -log_pt
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=1)
            ce_loss = (1.0 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha[targets]
        return loss.mean()
