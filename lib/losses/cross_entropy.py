import torch
import torch.nn as nn

from torch import Tensor


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes
    """

    def __init__(self, epsilon=.1):
        super().__init__()

        self.epsilon = epsilon

        self.logsm = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: Tensor, targets: Tensor):
        num_classes = inputs.size(-1)

        log_probs = nn.LogSoftmax(dim=-1)(inputs)

        targets = torch.zeros_like(log_probs, device=inputs.device).scatter_(
            -1, targets.unsqueeze(-1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(dim=0).sum()

        return loss
