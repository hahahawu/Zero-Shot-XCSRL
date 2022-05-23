import argparse
import os, math
import logging, warnings

from pytorch_lightning import Trainer, seed_everything, callbacks
from torch.utils.data import DataLoader

from config import configuration
from dataset import CSRLData, DialogSet, CrossLingualSet, SRLSet
from processor import Processor
from models.base_model import BaseModel
from utils import save_hparams_dict, str2bool, load_from_pretrained_weights
from models.xsrl_model import XSRLModel
from models.dialogue_model import DialogueModel
from models.alignment_model import AlignmentModel
from models.srl_model import SRLModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_name", type=str, required=True)
    arg_parser.add_argument("--task", type=str, choices=["csrl", "dialogue", "language_model", "srl"], default='csrl')
    arg_parser.add_argument("--seed", type=int, default=42)

    # CSRL data path
    arg_parser.add_argument("--csrl_train_data_path", type=str, default=configuration["csrl_train_data_path"])
    arg_parser.add_argument("--csrl_dev_data_path", type=str, default=configuration["csrl_dev_data_path"])
    arg_parser.add_argument("--csrl_label_vocab_path", type=str, default=configuration["csrl_label_vocab_path"])

    # few-shot setting
    arg_parser.add_argument("--few_shot", type=int, default=None)
    arg_parser.add_argument("--en_fs", action="store_true", default=False)

    # dialogue data path
    arg_parser.add_argument("--dialogue_train_path", type=list, default=configuration["dialogue_train_path"])
    arg_parser.add_argument("--dialogue_dev_path", type=list, default=configuration["dialogue_dev_path"])

    # language model data path
    arg_parser.add_argument("--lm_alignment_train_path", type=str, default=configuration["lm_alignment_train_path"])
    arg_parser.add_argument("--lm_alignment_dev_path", type=str, default=configuration["lm_alignment_dev_path"])
    arg_parser.add_argument("--lm_monolingual_train_path", type=str, default=configuration["lm_monolingual_train_path"])
    arg_parser.add_argument("--lm_monolingual_dev_path", type=str, default=configuration["lm_monolingual_dev_path"])

    # SRL data path
    arg_parser.add_argument("--srl_train_path", type=list, default=configuration["srl_train_path"])
    arg_parser.add_argument("--srl_dev_path", type=list, default=configuration["srl_dev_path"])
    arg_parser.add_argument("--srl_lang", choices=['zh', 'en', 'zh+en'], default='zh')

    # training config
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--num_workers", type=int, default=0)
    arg_parser.add_argument("--fp16", action="store_true", default=False)
    arg_parser.add_argument("--gpus", type=int, default=0)
    arg_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    arg_parser.add_argument("--patience", type=int, default=10)
    arg_parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    arg_parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    arg_parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    arg_parser.add_argument("--accelerator", type=str, default=None)
    arg_parser.add_argument("--auto_find_lr", type=str2bool, default=False)
    arg_parser.add_argument("--auto_find_batch_size", type=str2bool, default=False)
    arg_parser.add_argument("--max_sequence_length", type=int, default=512)
    arg_parser.add_argument("--frozen_layer_names", type=str, default=None)
    arg_parser.add_argument("--selection_mode", type=str, choices=["min", "max"], default='max')

    # epochs and steps
    arg_parser.add_argument("--max_epochs", type=int, default=100)
    arg_parser.add_argument("--min_epochs", type=int, default=5)
    arg_parser.add_argument("--max_steps", type=int, default=10000)
    arg_parser.add_argument("--min_steps", type=int, default=1000)

    # limited data for training, validation and test
    arg_parser.add_argument("--limit_train_batches", type=float, default=1.0)
    arg_parser.add_argument("--limit_val_batches", type=float, default=1.0)
    arg_parser.add_argument("--limit_test_batches", type=float, default=1.0)
    arg_parser.add_argument("--limit_predict_batches", type=float, default=1.0)

    # check val and log
    arg_parser.add_argument("--val_check_interval", type=float, default=1.0)
    arg_parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    arg_parser.add_argument("--log_every_n_steps", type=int, default=10)
    arg_parser.add_argument("--num_sanity_val_steps", type=int, default=2)

    # resume from dialogue pretraining
    arg_parser.add_argument("--pretrained_encoder", type=str, default=None)

    # language model pretraining
    arg_parser.add_argument("--training_strategy", choices=["alignment", "monolingual"], default='alignment')
    arg_parser.add_argument("--train_psi", action="store_true", default=False)

    # dialogue model pretraining
    arg_parser.add_argument("--is_reorder", action="store_true", default=False)
    arg_parser.add_argument("--is_role_detection", action="store_true", default=False)

    # fine tune configuration
    arg_parser.add_argument("--frozen_steps", type=int, default=-1)
    arg_parser.add_argument("--frozen_epochs", type=float, default=-1)
    arg_parser.add_argument("--unfreeze_layer_names", type=str, default=None)
    arg_parser.add_argument("--permanent_frozen_layers", type=str, default=None)

    # add model-specific args
    arg_parser = BaseModel.add_model_specific_args(arg_parser)

    # add all trainer options to args
    hparams = arg_parser.parse_args()
    seed_everything(hparams.seed)

    if not os.path.exists(hparams.checkpoint_dir):
        os.mkdir(hparams.checkpoint_dir)
    model_dir = os.path.join(hparams.checkpoint_dir, hparams.model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if hparams.task == 'csrl':

        if hparams.few_shot is not None:
            train_data_path = configuration["{}_shot_csrl_data".format(hparams.few_shot)]
        elif hparams.en_fs:
            train_data_path = configuration["few_shot_csrl_en_data"]
            hparams.csrl_dev_data_path = train_data_path
        else:
            train_data_path = hparams.csrl_train_data_path

        train_set = CSRLData(train_data_path)
        dev_set = CSRLData(hparams.csrl_dev_data_path)

        processor = Processor(task=hparams.task, pretrain_model_name=hparams.language_model,
                              label_path=hparams.csrl_label_vocab_path, max_seq_len=hparams.max_sequence_length)

        train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=hparams.num_workers, collate_fn=processor.csrl_collate_fn)
        dev_dataloader = DataLoader(dev_set, batch_size=hparams.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=hparams.num_workers, collate_fn=processor.csrl_collate_fn)

        # additional parameters
        hparams.steps_per_epoch = math.ceil(len(train_dataloader) / (hparams.batch_size *
                                                                     hparams.accumulate_grad_batches))
        hparams.num_labels = len(processor.label_vocab)
        hparams.vocab_size = len(processor.tokenizer)

        model = XSRLModel(hparams=hparams)

        if hparams.pretrained_encoder:
            load_from_pretrained_weights(model, hparams.pretrained_encoder)

    elif hparams.task == 'srl':
        train_set = SRLSet(hparams.srl_train_path, hparams.srl_lang)
        dev_set = SRLSet(hparams.srl_dev_path, hparams.srl_lang)

        processor = Processor(task=hparams.task, pretrain_model_name=hparams.language_model,
                              max_seq_len=hparams.max_sequence_length, label_path=hparams.csrl_label_vocab_path)
        train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=hparams.num_workers, collate_fn=processor.srl_collate_fn)
        dev_dataloader = DataLoader(dev_set, batch_size=hparams.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=hparams.num_workers, collate_fn=processor.srl_collate_fn)
        # additional parameters
        hparams.steps_per_epoch = math.ceil(len(train_dataloader) / (hparams.batch_size *
                                                                     hparams.accumulate_grad_batches))
        hparams.num_labels = len(processor.label_vocab)
        hparams.vocab_size = len(processor.tokenizer)

        model = SRLModel(hparams=hparams)

        if hparams.pretrained_encoder:
            load_from_pretrained_weights(model, hparams.pretrained_encoder)

    elif hparams.task == 'dialogue':
        train_set = DialogSet(hparams.dialogue_train_path)
        dev_set = DialogSet(hparams.dialogue_dev_path)

        processor = Processor(task=hparams.task, pretrain_model_name=hparams.language_model,
                              max_seq_len=hparams.max_sequence_length, is_reorder=hparams.is_reorder,
                              is_role_detection=hparams.is_role_detection)

        train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=hparams.num_workers, collate_fn=processor.dialog_collate_fn)
        dev_dataloader = DataLoader(dev_set, batch_size=hparams.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=hparams.num_workers, collate_fn=processor.dialog_collate_fn)
        # additional parameters
        hparams.steps_per_epoch = math.ceil(len(train_dataloader) / (hparams.batch_size *
                                                                     hparams.accumulate_grad_batches))
        hparams.vocab_size = len(processor.tokenizer)

        model = DialogueModel(hparams=hparams)

        if hparams.pretrained_encoder:
            load_from_pretrained_weights(model, hparams.pretrained_encoder)

    elif hparams.task == 'language_model':
        if hparams.training_strategy == 'alignment':
            train_set = CrossLingualSet(hparams.lm_alignment_train_path, use_aligned_data=True)
            dev_set = CrossLingualSet(hparams.lm_alignment_dev_path, use_aligned_data=True)
        elif hparams.training_strategy == 'monolingual':
            train_set = CrossLingualSet(hparams.lm_monolingual_train_path, False)
            dev_set = CrossLingualSet(hparams.lm_monolingual_dev_path, False)
        else:
            raise ValueError("Unexpected training strategy %s", hparams.training_strategy)

        processor = Processor(task=hparams.task, pretrain_model_name=hparams.language_model,
                              max_seq_len=hparams.max_sequence_length,
                              alignment=hparams.training_strategy == 'alignment')

        train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=hparams.num_workers, collate_fn=processor.lm_collate_fn)
        dev_dataloader = DataLoader(dev_set, batch_size=hparams.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=hparams.num_workers, collate_fn=processor.lm_collate_fn)

        hparams.steps_per_epoch = math.ceil(len(train_dataloader) / (hparams.batch_size *
                                                                     hparams.accumulate_grad_batches))
        hparams.num_labels = len(processor.word_vocab)
        hparams.vocab_size = len(processor.tokenizer)

        model = AlignmentModel(hparams=hparams)

        if hparams.pretrained_encoder:
            load_from_pretrained_weights(model, hparams.pretrained_encoder)

    else:
        raise ValueError("Unexpected task {}.".format(hparams.task))

    model_checkpoint_path = os.path.join(model_dir, hparams.task + ".{epoch}.{select_metrics:0.4f}")
    hparams_config_path = os.path.join(model_dir, "h_params.json")
    save_hparams_dict(hparams, hparams_config_path)
    processor.tokenizer.save_pretrained(model_dir)

    early_stop_callback = callbacks.EarlyStopping(
        monitor='select_metrics',
        patience=hparams.patience,
        verbose=True,
        min_delta=0.0002,
        mode=hparams.selection_mode,
        strict=True
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename=model_checkpoint_path,
        monitor='select_metrics',
        verbose=True,
        save_last=False,
        mode=hparams.selection_mode
    )

    lr_logger = callbacks.LearningRateMonitor(logging_interval="epoch")
    train_callbacks = [lr_logger, early_stop_callback, checkpoint_callback]

    trainer = Trainer(
        logger=True,
        checkpoint_callback=True,
        auto_select_gpus=False,
        auto_lr_find=hparams.auto_find_lr,
        auto_scale_batch_size=hparams.auto_find_batch_size,
        callbacks=train_callbacks,
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        progress_bar_refresh_rate=None,
        overfit_batches=0.0,
        track_grad_norm=-1,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        max_steps=hparams.max_steps,
        min_steps=hparams.min_steps,
        max_time=None,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        limit_predict_batches=hparams.limit_predict_batches,
        val_check_interval=hparams.val_check_interval if hparams.val_check_interval <= 1.0 else
        int(hparams.val_check_interval),
        log_every_n_steps=hparams.log_every_n_steps,
        accelerator=hparams.accelerator,
        precision=16 if hparams.fp16 else 32,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        deterministic=False,
        reload_dataloaders_every_epoch=False,
        terminate_on_nan=True,
        amp_backend='native',
        amp_level='02',
        num_sanity_val_steps=hparams.num_sanity_val_steps
    )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=dev_dataloader)
