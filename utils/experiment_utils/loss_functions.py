import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "FocalLoss",
]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(dim=1, index=targets.long().unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha[targets.long()]
        return loss.mean()
