from typing import Any
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from layers.state_encoder import StateEncoder
from utils import Embedding
from torch_scatter import scatter_mean


class XSRLModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        h_params = kwargs["hparams"]

        hidden_size = h_params.hidden_size

        feature_dim = 1
        self.predicate_embedding = Embedding(num_embeddings=3, embedding_dim=hidden_size, padding_idx=0)
        if h_params.use_turn_info:
            self.turn_embedding = Embedding(num_embeddings=12, embedding_dim=hidden_size, padding_idx=0)
            feature_dim += 1
        if h_params.use_role_info:
            self.role_embedding = Embedding(num_embeddings=3, embedding_dim=hidden_size, padding_idx=0)
            feature_dim += 1

        self.feature_combiner = StateEncoder(
            input_size=feature_dim * hidden_size,
            state_size=hidden_size // 2,
            output_size=hidden_size,
            num_layers=2,
            activation=h_params.activation_fn,
            dropout_rate=h_params.dropout
        )

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
        self.predicate_argument_encoder, pred_arg_output_size = self.get_encoder(h_params.pa_encoder_type,
                                                                                 input_size=hidden_size,
                                                                                 hidden_size=hidden_size,
                                                                                 mha_heads=h_params.mha_heads,
                                                                                 mha_num_layers=h_params.mha_num_layers,
                                                                                 mha_dropout=h_params.mha_dropout,
                                                                                 lstm_num_layers=h_params.lstm_num_layers,
                                                                                 lstm_dropout=h_params.lstm_dropout,
                                                                                 lstm_bidirectional=h_params.lstm_bidirectional)

        self.hierarchical_feature_combiner = StateEncoder(
            input_size=utterance_output_size + dialogue_output_size + hidden_size,
            state_size=hidden_size,
            num_layers=1,
            activation=h_params.activation_fn,
            dropout_rate=h_params.dropout
        )

        self.csrl_argument_classifier = torch.nn.Linear(self.predicate_argument_encoder.sequence_state_size,
                                                        h_params.num_labels)

    def encode_dialogue(self, features, x):
        sequence_length = x["sequence_length"]
        dialogue_indices = x["dialogue_indices"]  # bsz, seq
        utterance_num = x["utterance_num"]

        bsz, max_utterance_num = dialogue_indices.shape

        # bsz, seq, dim
        utterance_level_feature = self.utterance_encoder(features, sequence_length)
        dialogue_level_feature = scatter_mean(utterance_level_feature, index=dialogue_indices, dim=1)

        # bsz, utt_num, dim
        dialogue_level_feature = dialogue_level_feature[:, 1:max_utterance_num + 1:, :]
        dialogue_level_feature = self.dialogue_encoder(dialogue_level_feature, utterance_num)

        # add zero vector as paddings in the front of the features
        dim = dialogue_level_feature.shape[-1]
        context_embeddings = torch.cat([torch.zeros((bsz, 1, dim), device=self.device),
                                        dialogue_level_feature], dim=1)
        # mapping the dialogue features back wth utterance features
        # bsz, seq, dim
        context_embeddings = torch.gather(context_embeddings, dim=1,
                                          index=dialogue_indices.unsqueeze(-1).expand(-1, -1, dim))

        context_embeddings = torch.cat([utterance_level_feature, context_embeddings], dim=-1)

        return context_embeddings

    def forward(self, x):
        predicate_ids = x["predicate_ids"]
        turn_ids = x["turn_ids"]
        role_ids = x["speaker_role_ids"]

        # bsz, seq, dim
        _, word_embeddings = self.encode_sent(x)
        predicate_embeddings = self.predicate_embedding(predicate_ids)

        feature_list = [word_embeddings]
        if self.h_params.use_turn_info:
            turn_embeddings = self.turn_embedding(turn_ids)
            feature_list.append(turn_embeddings)
        if self.h_params.use_role_info:
            role_embeddings = self.role_embedding(role_ids)
            feature_list.append(role_embeddings)

        combined_feature_embeddings = torch.cat(feature_list, dim=-1)
        combined_feature_embeddings = self.feature_combiner(combined_feature_embeddings)
        context_embeddings = self.encode_dialogue(combined_feature_embeddings, x)

        context_embeddings = torch.cat([context_embeddings, predicate_embeddings], dim=-1)
        context_embeddings = self.hierarchical_feature_combiner(context_embeddings)
        final_repr = self.predicate_argument_encoder(context_embeddings, x["sequence_length"])
        argument_logits = self.csrl_argument_classifier(final_repr)

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
        device = self.device
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

        token_mask = torch.arange(max_seq_length).unsqueeze(0).to(device) < sequence_lengths.unsqueeze(1)
        token_mask = torch.tensor(token_mask).float()
        none_zero_mask = torch.gt(prediction, 0).float().to(device) * token_mask.contiguous().view(-1)
        eq_num = torch.eq(prediction, label).float().to(device)
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

    def __prepare_test_data(self, data_example):
        word_ids = data_example["word_ids"].long().to(self.device)
        predicate_ids = data_example["predicate_ids"].long().to(self.device)
        turn_ids = data_example["turn_ids"].long().to(self.device)
        subword_indices = data_example["subword_indices"].long().to(self.device)
        sequence_length = torch.tensor(data_example["sequence_length"]).long().to(self.device)
        tokenized_sequence_length = torch.tensor(data_example["tokenized_sequence_length"]).long().to(self.device)
        dialogue_indices = data_example["dialogue_indices"].long().to(self.device)
        utterance_num = torch.tensor(data_example["utterance_num"]).long().to(self.device)
        speaker_role_ids = data_example["speaker_role_ids"].long().to(self.device)

        if len(word_ids.shape) == 1:
            word_ids = word_ids.view(1, -1)
            predicate_ids = predicate_ids.view(1, -1)
            turn_ids = turn_ids.view(1, -1)
            subword_indices = subword_indices.view(1, -1)
            sequence_length = sequence_length.view(-1)
            tokenized_sequence_length = tokenized_sequence_length.view(-1)
            dialogue_indices = dialogue_indices.view(1, -1)
            utterance_num = utterance_num.view(-1)
            speaker_role_ids = speaker_role_ids.view(1, -1)

        return {
            "word_ids": word_ids,
            "predicate_ids": predicate_ids,
            "turn_ids": turn_ids,
            "speaker_role_ids": speaker_role_ids,
            "subword_indices": subword_indices,
            "sequence_length": sequence_length,
            "tokenized_sequence_length": tokenized_sequence_length,
            "dialogue_indices": dialogue_indices,
            "utterance_num": utterance_num
        }

    def predict(self, data_example, id2label, need_mapping=True):
        batch_sample = self.__prepare_test_data(data_example)
        bsz = batch_sample["word_ids"].shape[0]
        with torch.no_grad():
            csrl_logits = self(batch_sample)
            _, predictions = torch.max(F.softmax(csrl_logits, dim=-1), dim=-1)
            predictions = predictions.contiguous().view(bsz, -1).cpu().numpy()
            valid_predictions = []
            for i in range(bsz):
                sequence_length = batch_sample["sequence_length"][i].item()
                prediction = predictions[i][:sequence_length]
                prediction = [id2label[idx] for idx in prediction][1:-1]
                valid_predictions.append(prediction)

            if need_mapping:
                # [CLS] SENT [SEP]
                subword_indices = batch_sample["subword_indices"].tolist()[0][1:-1]
                subword_indices = [w - 2 for w in subword_indices]

                return predictions, subword_indices

            return valid_predictions

    @staticmethod
    def add_model_specific_args(parent_parser):
        return BaseModel.add_model_specific_args(parent_parser)
