import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        embed_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        if vocab_size == embed_size:
            self.embed = None
        else:
            self.embed = nn.Linear(vocab_size, embed_size)

        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            bias=False,
        )
        self.out_channels = hidden_dim * 2 if bidirectional else hidden_dim

        self._init_weight()

    def forward(self, captions):
        text = torch.stack([caption.text for caption in captions], dim=1)
        text_length = torch.stack(
            [caption.length for caption in captions], dim=1)

        text_length = text_length.view(-1)
        text = text.view(-1, text.size(-1))  # b x l

        if self.embed is not None:
            text = self.embed(text)

        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

    def gru_out(self, embed, text_length):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack)
        gru_sort_out, *_ = nn.utils.rnn.pad_packed_sequence(
            gru_sort_out, batch_first=True)  # Added '_'

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        return gru_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


def build_gru(hidden_dim: int, vocab_size: int, embed_size: int, num_layer: int, dropout: float, bidirectional: bool,  freeze_model: bool):
    model = GRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        dropout,
        bidirectional,
    )

    if freeze_model:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
