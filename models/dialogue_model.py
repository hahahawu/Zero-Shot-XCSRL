from typing import Any, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from layers.state_encoder import StateEncoder

from models.base_model import BaseModel
from torch_scatter import scatter_mean


class DialogueModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        h_params = kwargs["hparams"]
        hidden_size = h_params.hidden_size

        self.h_params = h_params
        self.save_hyperparameters(h_params)

        self.utterance_encoder, utterance_output_size = self.get_encoder(h_params.utterance_encoder_type,
                                                                         input_size=hidden_size,
                                                                         hidden_size=hidden_size,
                                                                         mha_heads=h_params.mha_heads,
                                                                         mha_num_layers=h_params.mha_num_layers,
                                                                         mha_dropout=h_params.mha_dropout,
                                                                         lstm_num_layers=h_params.lstm_num_layers,
                                                                         lstm_dropout=h_params.lstm_dropout,
                                                                         lstm_bidirectional=h_params.lstm_bidirectional)
        self.dialogue_encoder, dialogue_output_size = self.get_encoder(h_params.dialogue_encoder_type,
                                                                       input_size=utterance_output_size,
                                                                       hidden_size=utterance_output_size,
                                                                       mha_heads=h_params.mha_heads,
                                                                       mha_num_layers=h_params.mha_num_layers,
                                                                       mha_dropout=h_params.mha_dropout,
                                                                       lstm_num_layers=h_params.lstm_num_layers,
                                                                       lstm_dropout=h_params.lstm_dropout,
                                                                       lstm_bidirectional=h_params.lstm_bidirectional)

        self.dialogue_feature_proj = StateEncoder(
            input_size=dialogue_output_size,
            output_size=hidden_size,
            state_size=hidden_size,
            num_layers=1,
            activation='swish',
            dropout_rate=h_params.dropout,
        )

        self.is_reorder = h_params.is_reorder
        self.is_role_detection = h_params.is_role_detection

        if self.is_reorder:
            self.turn_decoder = torch.nn.Linear(hidden_size, 11, bias=False)
        if self.is_role_detection:
            self.speaker_decoder = torch.nn.Linear(hidden_size, 2, bias=False)
        torch.nn.init.normal_(self.turn_decoder.weight, mean=0, std=hidden_size ** -0.5)
        torch.nn.init.normal_(self.speaker_decoder.weight, mean=0, std=hidden_size ** -0.5)

    def forward(self, x):
        _, word_embeddings = self.word_encoder(x["input_ids"], subword_indices=x["subword_indices"],
                                               sequence_lengths=x["tokenized_sequence_length"])
        max_sequence_length = max(x["sequence_length"])
        word_embeddings = word_embeddings[:, 1:max_sequence_length + 1, :]

        utterance_feature = self.utterance_encoder(word_embeddings, x["sequence_length"])
        gathered_utterance_feature = scatter_mean(utterance_feature, index=x["dialogue_indices"], dim=1)

        max_utterance_num = max(x["utterance_num"])
        gathered_utterance_feature = gathered_utterance_feature[:, 1:max_utterance_num + 1, :]
        dialogue_feature = self.dialogue_encoder(gathered_utterance_feature, x["utterance_num"])
        dialogue_feature = self.dialogue_feature_proj(dialogue_feature)

        return dialogue_feature

    def __shared_step(self, batch):
        reorder_loss, role_loss = 0, 0
        reorder_logits, role_logits = None, None
        if self.is_reorder:
            dialogue_feature = self({
                "input_ids": batch["reorder_input_ids"],
                "subword_indices": batch["reorder_subword_indices"],
                "tokenized_sequence_length": batch["reorder_tokenized_sequence_length"],
                "sequence_length": batch["reorder_sequence_length"],
                "position_ids": batch["reorder_position_ids"],
                "dialogue_indices": batch["reorder_dialogue_indices"],
                "utterance_num": batch["reorder_utterance_num"]
            })
            reorder_logits = self.turn_decoder(dialogue_feature)
            reorder_loss = F.cross_entropy(
                reorder_logits.view(-1, 11),
                batch["reorder_label"].view(-1),
                ignore_index=-100,
                reduction='mean'
            )

        if self.is_role_detection:
            dialogue_feature = self({
                "input_ids": batch["role_input_ids"],
                "subword_indices": batch["role_subword_indices"],
                "tokenized_sequence_length": batch["role_tokenized_sequence_length"],
                "sequence_length": batch["role_sequence_length"],
                "position_ids": batch["role_position_ids"],
                "dialogue_indices": batch["role_dialogue_indices"],
                "utterance_num": batch["role_utterance_num"]
            })
            role_logits = self.speaker_decoder(dialogue_feature)
            role_loss = F.cross_entropy(
                role_logits.view(-1, 2),
                batch["role_label"].view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        loss = reorder_loss + role_loss
        return loss, reorder_loss, role_loss, (reorder_logits, role_logits)

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        loss, reorder_loss, role_loss, _ = self.__shared_step(batch)

        self.log("model_lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)
        self.log("lm_lr", self.optimizers().param_groups[1]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)

        return loss

    def validation_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        loss, reorder_loss, role_loss, (reorder_logits, role_logits) = self.__shared_step(batch)

        _, reorder_prediction = torch.max(F.softmax(reorder_logits, dim=-1), dim=-1)
        _, role_prediction = torch.max(F.softmax(role_logits, dim=-1), dim=-1)

        reorder_eq_num = torch.eq(reorder_prediction.view(-1), batch["reorder_label"].view(-1)).sum().float().to(
            self.device)
        reorder_total_num = torch.gt(batch["reorder_label"].view(-1), -1).sum().float()

        role_eq_num = torch.eq(role_prediction.view(-1), batch["role_label"].view(-1)).sum().float().to(self.device)
        role_total_num = torch.gt(batch["role_label"].view(-1), -1).sum().float()

        return {
            "loss": loss,
            "reorder_loss": reorder_loss,
            "role_loss": role_loss,
            "reorder_eq_num": reorder_eq_num,
            "reorder_total_num": reorder_total_num,
            "role_eq_num": role_eq_num,
            "role_total_num": role_total_num
        }

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_reorder_loss = torch.stack([x["reorder_loss"] for x in outputs]).mean()
        avg_role_loss = torch.stack([x["role_loss"] for x in outputs]).mean()

        total_reorder_eq_num = torch.stack([x["reorder_eq_num"] for x in outputs]).sum()
        total_reorder_total_num = torch.stack([x["reorder_total_num"] for x in outputs]).sum()
        total_role_eq_num = torch.stack([x["role_eq_num"] for x in outputs]).sum()
        total_role_total_num = torch.stack([x["role_total_num"] for x in outputs]).sum()

        reorder_acc = total_reorder_eq_num / total_reorder_total_num if total_reorder_total_num != 0 else 0
        role_acc = total_role_eq_num / total_role_total_num if total_role_total_num != 0 else 0

        select_metrics = avg_loss

        print("========================")
        print("avg loss: {}".format(avg_loss))
        print("avg_reorder_loss: {}".format(avg_reorder_loss))
        print("avg_role_loss: {}".format(avg_role_loss))
        print("reorder accuracy: {}".format(reorder_acc))
        print("role accuracy: {}".format(role_acc))
        print("========================")

        self.log("select_metrics", torch.tensor(select_metrics, device=self.device), prog_bar=True, logger=False,
                 on_epoch=True, on_step=False, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return BaseModel.add_model_specific_args(parent_parser)
