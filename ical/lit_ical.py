import zipfile
from typing import List
import editdistance
import json
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from ical.datamodule import Batch, vocab
from ical.model.ical import ICAL
from ical.utils.utils import (
    ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out, plicit_tgt_out)


class LitICAL(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        milestones: List[int] = [40, 55],
        dynamic_weight: bool = True,
        vocab_size: int = 114,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.ical_model = ICAL(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.ical_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        fusion_tgt, fusion_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False)
        exp_tgt, exp_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False)
        implicit_tgt, implicit_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=True)

        exp_out_hat, imp_out_hat, fusion_out_hat = self(
            batch.imgs, batch.mask, exp_tgt)

        exp_loss = ce_loss(exp_out_hat, exp_out)
        implicit_loss = ce_loss(imp_out_hat, implicit_out,
                                need_weight=self.hparams.dynamic_weight)
        fusion_loss = ce_loss(fusion_out_hat, fusion_out)

        self.log("train_implicit_loss", implicit_loss,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_explicit_loss", exp_loss,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_fusion_loss", fusion_loss,
                 on_step=False, on_epoch=True, sync_dist=True)

        loss = exp_loss + implicit_loss + fusion_loss
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        fusion_tgt, fusion_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False)
        exp_tgt, exp_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False)
        implicit_tgt, implicit_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=True)

        exp_out_hat, imp_out_hat, fusion_out_hat = self(
            batch.imgs, batch.mask, exp_tgt)

        exp_loss = ce_loss(exp_out_hat, exp_out)
        implicit_loss = ce_loss(imp_out_hat, implicit_out,
                                need_weight=self.hparams.dynamic_weight)
        fusion_loss = ce_loss(fusion_out_hat, fusion_out)

        self.log(
            "val_fusion_loss",
            fusion_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_exp_loss",
            exp_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_imp_loss",
            implicit_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        gts = [vocab.indices2words(ind) for ind in batch.indices]
        preds = [vocab.indices2words(h.seq) for h in hyps]

        implicit_pred_tokens = self.gen_implicit_tokens(batch, hyps)
        im_pred_tokens = [vocab.indices2words(
            tokens) for tokens in implicit_pred_tokens]

        return batch.img_bases, preds, gts, im_pred_tokens

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        errors_dict = {}
        predictions_dict = {}
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, gts, im_pred_tokens in test_outputs:
                for img_base, pred, gt, im_pred_token in zip(img_bases, preds, gts, im_pred_tokens):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
                    distance = editdistance.eval(pred, gt)
                    if distance > 0:
                        errors_dict[img_base] = {
                            "pred": " ".join(pred),
                            "gt": " ".join(gt),
                            "dist": distance,
                            "im_tokens": " ".join(im_pred_token),
                        }

                    predictions_dict[img_base] = {
                        "pred": " ".join(pred),
                        "gt": " ".join(gt),
                        "dist": distance,
                        "im_tokens": " ".join(im_pred_token),
                    }
        with open("errors.json", "w") as f:
            json.dump(errors_dict, f)
        with open("predictions.json", "w") as f:
            json.dump(predictions_dict, f)

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.ical_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def gen_implicit_tokens(self, batch: Batch, hyps: List[Hypothesis]):
        hyps_list = [h.seq for h in hyps]
        fusion_tgt, fusion_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=False)
        exp_tgt, exp_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=False)
        implicit_tgt, implicit_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=True)

        exp_out_hat, imp_out_hat, fusion_out_hat = self(
            batch.imgs, batch.mask, exp_tgt)
        _, max_indices = torch.max(imp_out_hat, dim=2)
        max_indices = max_indices[:4, :].tolist()
        return max_indices
