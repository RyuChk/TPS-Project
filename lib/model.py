import torch.nn as nn
from torch import Tensor

from backbones import build_resnet, build_gru
from embedding import Embed

# TODO: fill defaults


class Net(nn.Module):
    def __init__(self, model_arch: str, *,
                 visual_size: int,
                 textual_size: int,
                 hidden_dim: int,
                 vocab_size: int,
                 embed_size: int,
                 num_layer: int,
                 dropout: float,
                 bidirectional: bool,
                 res5_stride=2,
                 res5_dilation=1,
                 pretrained: str | None = None,
                 freeze_model: bool = False):
        super().__init__()

        self.visual_model = build_resnet(
            model_arch, res5_stride, res5_dilation, pretrained, freeze_model)
        self.textual_model = build_gru(
            hidden_dim, vocab_size, embed_size, num_layer, dropout, bidirectional, freeze_model)
        self.embed = Embed(visual_size, textual_size)

    def forward(self, images: Tensor, captions):
        visual_out = self.visual_model(images)
        textual_out = self.textual_model(captions)

        embed_out = self.embed(visual_out, textual_out)

        return embed_out
