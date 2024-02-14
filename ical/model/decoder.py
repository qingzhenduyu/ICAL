from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from ical.datamodule import vocab
from ical.model.pos_enc import WordPosEnc
from ical.model.transformer.arm import AttentionRefinementModule
from ical.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from ical.utils.generation_utils import DecodeModel


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(
            nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.SCCM = SCCM(d_model)
        self.fusion = FusionModule(d_model)
        self.exp_proj = nn.Linear(d_model, vocab_size)
        self.imp_proj = nn.Sequential(
            nn.ReLU(), nn.Linear(d_model, vocab_size))
        self.fusion_proj = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(d_model, vocab_size))

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        exp_out = rearrange(out, "l b d -> b l d")
        imp_out = self.SCCM(exp_out, tgt_mask, tgt_pad_mask)

        fusion_out = self.fusion(exp_out, imp_out)
        exp_out = self.exp_proj(exp_out)
        imp_out = self.imp_proj(imp_out)
        fusion_out = self.fusion_proj(fusion_out)

        return exp_out, imp_out, fusion_out

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        exp_out, imp_out, fusion_out = self(src[0], src_mask[0], input_ids)
        return fusion_out

class SCCM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=8, dim_feedforward=1024, dropout=0.3,
            ),
            num_layers=1,
        )

    def forward(self, out: FloatTensor, tgt_mask: LongTensor, src_key_padding_mask: LongTensor):
        """generate implicit logits

        Parameters
        ----------
        out : FloatTensor
            [b, l, d]
        tgt_mask: LongTensor
            [b, l]
        src_key_padding_mask: LongTensor
            [b, l, d]
        Returns
        -------
        FloatTensor
            [b, l, d]
        """
        out = rearrange(out, "b t d -> t b d")
        out = self.te(
            src=out, mask=tgt_mask, src_key_padding_mask=src_key_padding_mask
        )
        out = rearrange(out, "t b d -> b t d")

        return out


class FusionModule(nn.Module):
    def __init__(self,  d_model: int,):
        super(FusionModule, self).__init__()
        self.d_model = d_model
        self.w_att = nn.Linear(2 * d_model, d_model)

    def forward(self, e_feature: FloatTensor, i_feature: FloatTensor):
        """generate output fusing e_feature & i_feature

        Parameters
        ----------
        e_feature : FloatTensor
            [b, l, d]
        i_feature: FloatTensor
            [b, l, d]

        Returns
        -------
        FloatTensor
            [b, l, d]
        """
        f = torch.cat((e_feature, i_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * i_feature + (1 - f_att) * e_feature
        return output
