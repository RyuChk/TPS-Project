import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .cross_entropy import CrossEntropyLabelSmooth


def instance_loss(projection: Tensor, visual_embed: Tensor, textual_embed: Tensor, labels: Tensor, *,
                  scale=1.,
                  norm=False,
                  epsilon=.0):
    if norm:
        visual_embed = F.normalize(visual_embed, dim=-1)
        textual_embed = F.normalize(textual_embed, dim=-1)

    projection = F.normalize(projection, dim=0)

    visual_logits = scale * torch.matmul(visual_embed, projection)
    textual_logits = scale * torch.matmul(textual_embed, projection)

    if epsilon > 0.:
        criterion = CrossEntropyLabelSmooth(epsilon)
    else:
        criterion = nn.CrossEntropyLoss()

    loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)

    return loss


def global_align_loss(visual_embed: Tensor, textual_embed: Tensor, labels: Tensor, *,
                      alpha=.6,
                      beta=.4,
                      scale_pos=10.,
                      scale_neg=40.):
    batch_size = labels.size(0)
    visual_norm = F.normalize(visual_embed, dim=1)
    textual_norm = F.normalize(textual_embed, dim=1)

    similarity = torch.matmul(visual_norm, textual_norm.T)
    labels = labels.expand(batch_size, batch_size).eq(
        labels.expand(batch_size, batch_size).t()).float()

    pos_inds = labels == 1
    neg_inds = labels == 0

    loss_pos = torch.log(
        1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(
        1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))

    loss = (loss_pos.sum() + loss_neg.sum()) * 2.

    loss /= batch_size
    return loss
