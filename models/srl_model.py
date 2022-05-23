from typing import Any
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from utils import Embedding


class SRLModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        h_params = kwargs["hparams"]

        hidden_size = h_params.hidden_size
        self.srl_lang = h_params.srl_lang

        self.predicate_embedding = Embedding(num_embeddings=3, embedding_dim=hidden_size, padding_idx=0)

        self.utterance_encoder, utterance_output_size = self.get_encoder(h_params.utterance_encoder_type,
                                                                         input_size=hidden_size,
                                                                         hidden_size=hidden_size,
                                                                         mha_heads=h_params.mha_heads,
                                                                         mha_num_layers=h_params.mha_num_layers,
                                                                         mha_dropout=h_params.mha_dropout,
                                                                         lstm_num_layers=h_params.lstm_num_layers,
                                                                         lstm_dropout=h_params.lstm_dropout,
                                                                         lstm_bidirectional=h_params.lstm_bidirectional)
        self.predicate_argument_encoder, pa_output_size = self.get_encoder(h_params.pa_encoder_type,
                                                                           input_size=hidden_size,
                                                                           hidden_size=hidden_size,
                                                                           mha_heads=h_params.mha_heads,
                                                                           mha_num_layers=h_params.mha_num_layers,
                                                                           mha_dropout=h_params.mha_dropout,
                                                                           lstm_num_layers=h_params.lstm_num_layers,
                                                                           lstm_dropout=h_params.lstm_dropout,
                                                                           lstm_bidirectional=h_params.lstm_bidirectional)
        self.srl_projector = torch.nn.Linear(utterance_output_size + hidden_size, hidden_size)

        self.argument_classifier = torch.nn.Linear(pa_output_size, h_params.num_labels)

    def forward(self, x):
        predicate_ids = x["predicate_ids"]

        # bsz, seq, dim
        _, word_embeddings = self.encode_sent(x)
        predicate_embeddings = self.predicate_embedding(predicate_ids)
        context_embeddings = self.utterance_encoder(word_embeddings, x["sequence_length"])
        context_embeddings = torch.cat([context_embeddings, predicate_embeddings], dim=-1)
        context_embeddings = self.srl_projector(context_embeddings)
        final_repr = self.predicate_argument_encoder(context_embeddings, x["sequence_length"])

        argument_logits = self.argument_classifier(final_repr)

        return argument_logits

    def training_step(self, batch, batch_index):
        argument_scores = self(batch)  # (bsz, seq, dim)
        argument_classification_loss = F.cross_entropy(
            argument_scores.view(-1, self.h_params.num_labels),
            batch["label"].view(-1),
            reduction='mean',
            ignore_index=-100
        )
        loss = argument_classification_loss
        self.log("model_lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)
        self.log("lm_lr", self.optimizers().param_groups[1]['lr'], prog_bar=True, logger=False, on_epoch=False,
                 on_step=True)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_index):
        sequence_lengths = batch["sequence_length"]
        argument_scores = self(batch)
        max_seq_length = argument_scores.shape[1]
        argument_classification_loss = F.cross_entropy(
            argument_scores.view(-1, self.h_params.num_labels),
            batch["label"].view(-1),
            reduction="mean",
            ignore_index=-100
        )
        val_loss = argument_classification_loss
        _, prediction = torch.max(F.softmax(argument_scores, dim=-1), dim=-1)

        label = batch["label"].contiguous().view(-1)
        prediction = prediction.contiguous().view(-1)

        token_mask = torch.arange(max_seq_length).unsqueeze(0).to(self.device) < sequence_lengths.unsqueeze(1)
        token_mask = torch.tensor(token_mask).float()
        none_zero_mask = torch.gt(prediction, 0).float().to(self.device) * token_mask.contiguous().view(-1)
        eq_num = torch.eq(prediction, label).float().to(self.device)
        eq_num = (eq_num * none_zero_mask.contiguous().view(-1)).sum().cpu()

        total_pred_count = torch.sum(none_zero_mask).cpu()
        total_gold_count = torch.sum(torch.gt(label, 0)).cpu()

        return {
            "loss": val_loss,
            "eq_num": eq_num,
            "total_pred_num": total_pred_count,
            "total_gold_num": total_gold_count
        }

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        total_pred_num = torch.stack([x["total_pred_num"] for x in outputs]).sum()
        total_gold_num = torch.stack([x["total_gold_num"] for x in outputs]).sum()
        correct_num = torch.stack([x["eq_num"] for x in outputs]).sum()

        val_p = correct_num / total_pred_num
        val_r = correct_num / total_gold_num
        val_f1 = 2 * val_p * val_r / (val_p + val_r)

        select_metrics = val_f1

        self.log("select_metrics", torch.tensor(select_metrics, device=self.device), logger=False, on_epoch=True,
                 on_step=False, prog_bar=True, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return BaseModel.add_model_specific_args(parent_parser)
