from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from models.base_model import BaseModel


class AlignmentModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        h_params = kwargs["hparams"]

        self.vocab_size = h_params.num_labels
        self.h_params = h_params
        hidden_size = h_params.hidden_size

        self.save_hyperparameters(h_params)

        self.alignment_decoder = nn.Linear(hidden_size, self.vocab_size)
        if h_params.train_psi:
            self.psi_decoder = nn.Linear(self.word_encoder.language_model.config.hidden_size, 2)

    def forward(self, x) -> Any:
        input_ids = x["input_ids"]
        sequence_length = x["sequence_length"]
        token_embeddings = self.word_encoder(input_ids, sequence_lengths=sequence_length)
        mlm_logits = self.alignment_decoder(token_embeddings)
        if self.h_params.train_psi:
            psi_input_ids = x["psi_pair"]
            psi_sequence_length = x["psi_sequence_length"]
            psi_logits = self.word_encoder(psi_input_ids, sequence_lengths=psi_sequence_length, get_pooled_output=True)
            psi_logits = self.psi_decoder(psi_logits)
            return mlm_logits, psi_logits
        else:
            return mlm_logits, None

    def training_step(self, batch, batch_index):
        mlm_logits, psi_logits = self(batch)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.vocab_size),
            batch["mlm_label"].view(-1),
            reduction='mean',
            ignore_index=-100
        )
        psi_loss = 0
        if self.h_params.train_psi:
            psi_loss = F.cross_entropy(
                psi_logits.view(-1, 2),
                batch["psi_label"].view(-1),
                reduction='mean',
            )

        self.log("model_lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)
        self.log("lm_lr", self.optimizers().param_groups[1]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)
        return mlm_loss + psi_loss

    def validation_step(self, batch, batch_index):
        mlm_logits, psi_logits = self(batch)
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.vocab_size),
            batch["mlm_label"].view(-1),
            reduction='mean',
            ignore_index=-100
        )
        psi_loss = 0
        if self.h_params.train_psi:
            psi_loss = F.cross_entropy(
                psi_logits.view(-1, 2),
                batch["psi_label"].view(-1),
                reduction='mean',
            )
        return {
            "loss": mlm_loss + psi_loss,
            "mlm_loss": mlm_loss,
            "psi_loss": torch.as_tensor(psi_loss).float()
        }

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mlm_loss = torch.stack([x["mlm_loss"] for x in outputs]).mean()
        avg_psi_loss = torch.stack([x["psi_loss"] for x in outputs]).mean()
        select_metrics = avg_loss

        print("\n======================\n")
        print("average mlm loss: {}".format(avg_mlm_loss))
        print("average psi loss: {}".format(avg_psi_loss))
        print("\n======================\n")

        self.log("select_metrics", torch.tensor(select_metrics, device=self.device), prog_bar=True, logger=False,
                 on_epoch=True, on_step=False, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return BaseModel.add_model_specific_args(parent_parser)
