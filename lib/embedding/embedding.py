import torch.nn as nn
from torch import Tensor


class Embed(nn.Module):
    """A linear projection for embedding textual and visual features

    Args:
        visual_size (int): size of the visual feature
        textual_size (int): size of the textual feature
        feature_size (int): the embedding size for the visual and textual features. Default ``512``
    """

    def __init__(self, visual_size: int, textual_size: int,
                 *,
                 embed_dim=512):
        super().__init__()

        self.visual_proj = nn.Linear(visual_size, embed_dim)
        self.textual_proj = nn.Linear(textual_size, embed_dim)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, visual_features: Tensor, textual_features: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            visual_features (Tensor): Visual features of shape (L, E_v) for unbatched input, or (N, L, E_v) when the input is batched
            textual_features (Tensor): Textual features of shape (L, E_t) for unbatched input, or (N, L, E_t) when the input is batched
            captions (Tensor): ...

        Outputs:
            - embeds - Visual and textual embedding of shape (L, E) when input is unbatched, or (N, L, E) when input is batched, where 
                       L is the target sequence length, N is the batch size, and E is the embedding dimension `embed_dim`
        """
        batch_size = visual_features.size(0)

        visual_embed = visual_features.view(batch_size, -1)
        textual_embed = textual_features.view(batch_size, -1)

        visual_embed = self.visual_proj(visual_embed)
        textual_embed = self.textual_proj(textual_embed)

        return visual_embed, textual_embed
