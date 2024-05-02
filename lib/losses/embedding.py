import torch
import torch.nn as nn

from torch import Tensor

import lib.losses as losses


class EmbeddingLoss(nn.Module):
    def __init__(self,
                 feature_size: int,
                 num_classes: int,
                 *,
                 epsilon=0.,
                 scale_pos=10.,
                 scale_neg=40.):

        self.projection = nn.Parameter(
            torch.randn(feature_size, num_classes),
            requires_grad=True
        )
        self.epsilon = epsilon
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

        nn.init.xavier_uniform_(self.projection.data)

    def forward(self, visual_embed: Tensor, textual_embed: Tensor, captions):
        labels = torch.stack([caption.get_field('id')
                             for caption in captions]).long()

        instance_loss = losses.instance_loss(
            self.projection, visual_embed, textual_embed, labels, epsilon=self.epsilon)
        global_align_loss = losses.global_align_loss(
            visual_embed, textual_embed, labels, scale_pos=self.scale_pos, scale_neg=self.scale_neg)

        return instance_loss, global_align_loss
