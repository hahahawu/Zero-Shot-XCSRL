import argparse
from typing import Any, Optional, Callable

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer

from layers.sequence_encoder import SequenceEncoder
from layers.word_encoder import WordEncoder

from utils import str2bool


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs: Any) -> None:
        super(BaseModel, self).__init__()
        h_params = kwargs["hparams"]
        self.h_params = h_params

        if hasattr(h_params, "vocab_size"):
            vocab_size = h_params.vocab_size
        else:
            vocab_size = None
        self.word_encoder = WordEncoder(language_model=h_params.language_model, fine_tune=h_params.fine_tune_lm,
                                        vocab_size=vocab_size, word_dropout=h_params.word_dropout,
                                        hidden_size=h_params.hidden_size)

    @staticmethod
    def get_encoder(encoder_type, **kwargs):
        if 'lstm' in encoder_type:
            encoder = SequenceEncoder(
                encoder_type=encoder_type,
                lstm_input_size=kwargs["input_size"],
                lstm_hidden_size=kwargs["hidden_size"],
                lstm_num_layers=kwargs["lstm_num_layers"],
                lstm_dropout=kwargs["lstm_dropout"],
                lstm_bidirectional=kwargs["lstm_bidirectional"],
                pack_sequences=True
            )
        elif encoder_type == 'connected_mha':
            encoder = SequenceEncoder(
                encoder_type=encoder_type,
                input_size=kwargs["input_size"],
                mha_num_heads=kwargs["mha_heads"],
                mha_num_layers=kwargs["mha_num_layers"],
                mha_dropout=kwargs["mha_dropout"]
            )
        else:
            raise ValueError("Unexpected encoder type {}".format(encoder_type))
        return encoder, encoder.sequence_state_size

    def encode_sent(self, x):
        word_ids = x["word_ids"]
        sequence_length = x["sequence_length"]
        subword_indices = x["subword_indices"]
        tokenized_sequence_length = x["tokenized_sequence_length"]

        max_sequence_length = max(sequence_length)

        # bsz, t_seq, dim
        token_embeddings, word_embeddings = self.word_encoder(word_ids, subword_indices, tokenized_sequence_length)
        # bsz, seq, dim
        word_embeddings = word_embeddings[:, 1:max_sequence_length + 1, :]

        return token_embeddings, word_embeddings

    def get_lr_scheduler(self, optimizer):
        if self.h_params.lr_scheduler is None or self.h_params.lr_scheduler == 'linear':
            return None
        if self.h_params.lr_scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.h_params.max_epochs, eta_min=1e-6)
            interval = 'epoch'
        elif self.h_params.lr_scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                                      min_lr=1e-6)
            interval = 'epoch'
        else:
            raise ValueError("Unexpected lr_scheduler: {}".format(self.h_params.lr_scheduler))
        scheduler_dict = {"scheduler": lr_scheduler, "interval": interval}
        if self.h_params.lr_scheduler == 'plateau':
            scheduler_dict["monitor"] = 'select_metrics'
        return scheduler_dict

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
                       optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None, on_tpu: bool = None,
                       using_native_amp: bool = None, using_lbfgs: bool = None) -> None:
        step = self.trainer.global_step
        if step == max(self.h_params.frozen_steps, self.h_params.steps_per_epoch * self.h_params.frozen_epochs):
            unfreeze_layers = self.h_params.unfreeze_layer_names.split(",") if \
                self.h_params.unfreeze_layer_names is not None else []
            permanent_frozen_layers = self.h_params.permanent_frozen_layers.split(",") if \
                self.h_params.permanent_frozen_layers is not None else []

            total_para, trainable_para, non_trainable_para = [], [], []
            for name, parameter in self.named_parameters():
                if any([x in name for x in permanent_frozen_layers]):
                    parameter.requires_grad = False
                elif any([_ in name for _ in unfreeze_layers]):
                    print("Parameters named {} UNFREEZE.".format(name))
                    parameter.requires_grad = True
                if parameter.requires_grad:
                    trainable_para.append(parameter.numel())
                else:
                    non_trainable_para.append(parameter.numel())
                total_para.append(parameter.numel())
            print("Trainable params: {}\nNon-trainable params: {}\nTotal params: {}\n".format(
                sum(trainable_para), sum(non_trainable_para), sum(total_para)))

        if self.h_params.lr_scheduler == 'linear':
            warmup_steps = self.h_params.warmup_epochs * self.h_params.steps_per_epoch
            training_steps = min(self.h_params.max_epochs * self.h_params.steps_per_epoch, self.h_params.max_steps)

            if step < warmup_steps:
                lr_scale = min(1., float(step + 1) / warmup_steps)
                optimizer.param_groups[0]['lr'] = lr_scale * self.h_params.model_lr
                optimizer.param_groups[1]['lr'] = lr_scale * self.h_params.lm_lr
                optimizer.param_groups[2]['lr'] = lr_scale * self.h_params.small_lr
            else:
                progress = float(step - warmup_steps) / float(max(1, training_steps - warmup_steps))
                lr_scale = (1. - progress)
                optimizer.param_groups[0]['lr'] = self.h_params.min_model_lr + lr_scale * (self.h_params.model_lr -
                                                                                           self.h_params.min_model_lr)
                optimizer.param_groups[1]['lr'] = self.h_params.min_lm_lr + lr_scale * (self.h_params.lm_lr -
                                                                                        self.h_params.min_lm_lr)
                optimizer.param_groups[2]['lr'] = self.h_params.min_small_lr + lr_scale * (self.h_params.small_lr -
                                                                                           self.h_params.min_small_lr)

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        base_params = []
        language_model_params = []
        no_decay_params = []
        small_lr_params = []
        no_decay = ["bias", "LayerNorm.weight"]

        frozen_layer_names = self.h_params.frozen_layer_names.split(",") if \
            self.h_params.frozen_layer_names is not None else []

        small_lr_group = self.h_params.small_lr_group.split(",") if self.h_params.small_lr_group is not None else []

        for name, parameter in self.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(parameter)
            else:
                if 'language_model' in name:
                    language_model_params.append(parameter)
                elif any([_ in name for _ in small_lr_group]):
                    small_lr_params.append(parameter)
                else:
                    base_params.append(parameter)
                if any([_ in name for _ in frozen_layer_names]):
                    print("Parameters named {} FROZEN.".format(name))
                    parameter.requires_grad = False

        optimizer = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': self.h_params.model_lr, 'weight_decay': self.h_params.weight_decay},
                {'params': language_model_params, 'lr': self.h_params.lm_lr,
                 'weight_decay': self.h_params.lm_weight_decay, 'correct_bias': False},
                {'params': small_lr_params, 'lr': self.h_params.small_lr, 'weight_decay': self.h_params.weight_decay},
                {'params': no_decay_params, 'lr': 2e-5, 'weight_decay': 0.0}
            ]
        )

        scheduler_dict = self.get_lr_scheduler(optimizer)

        if scheduler_dict:
            return [optimizer], [scheduler_dict]
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--language_model", type=str, default="xlm-roberta-base")
        parser.add_argument("--word_dropout", type=float, default=0.4)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--use_role_info", type=str2bool, default=False)
        parser.add_argument("--use_turn_info", type=str2bool, default=False)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--activation_fn", type=str, default='swish')

        # learning rate and weight decay
        parser.add_argument("--model_lr", type=float, default=5e-5)
        parser.add_argument("--min_model_lr", type=float, default=5e-5)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--lm_lr", type=float, default=5e-5)
        parser.add_argument("--min_lm_lr", type=float, default=5e-5)
        parser.add_argument("--lm_weight_decay", type=float, default=1e-4)
        parser.add_argument("--fine_tune_lm", type=str2bool, default=False)

        # warmup
        parser.add_argument("--warmup_epochs", type=float, default=0.0)
        parser.add_argument("--lr_scheduler", type=str, default=None)

        # Sequence LSTM
        parser.add_argument("--lstm_num_layers", type=int, default=1)
        parser.add_argument("--lstm_dropout", type=float, default=0.2)
        parser.add_argument("--lstm_bidirectional", type=str2bool, default=False)

        # self attention
        parser.add_argument("--mha_heads", type=int, default=4)
        parser.add_argument("--mha_num_layers", type=int, default=2)
        parser.add_argument("--mha_dropout", type=float, default=0.2)

        # encoder type
        parser.add_argument("--utterance_encoder_type", type=str, default='connected_lstm')
        parser.add_argument("--dialogue_encoder_type", type=str, default='connected_lstm')
        parser.add_argument("--pa_encoder_type", type=str, default='connected_lstm')

        # training with minimum lr
        parser.add_argument("--small_lr_group", type=str, default=None)
        parser.add_argument("--small_lr", type=float, default=5e-5)
        parser.add_argument("--min_small_lr", type=float, default=5e-5)

        return parser
