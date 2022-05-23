#!/bin/sh

for seed in 24 100 212 1027 12345;
do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --model_name v_CSRL+pretrained+seed_$seed \
  --task csrl \
  --gpus -1 \
  --seed $seed \
  --batch_size 48 \
  --num_workers 4 \
  --checkpoint_dir checkpoints \
  --patience 10 \
  --accumulate_grad_batches 1 \
  --max_epochs 100 \
  --min_epochs 5 \
  --max_steps 100000 \
  --min_steps 500 \
  --limit_train_batches 1.0 \
  --limit_val_batches 1.0 \
  --limit_test_batches 1.0 \
  --limit_predict_batches 1.0 \
  --val_check_interval 1.0 \
  --check_val_every_n_epoch 1 \
  --gradient_clip_val 1.0 \
  --language_model xlm-roberta-base \
  --use_turn_info 1 \
  --use_role_info 1 \
  --model_lr 4e-5 \
  --min_model_lr 2e-5 \
  --weight_decay 1e-6 \
  --lm_lr 4e-5 \
  --min_lm_lr 2e-5 \
  --lm_weight_decay 1e-6 \
  --fine_tune_lm 1 \
  --warmup_epochs 0.0 \
  --mha_num_layers 2 \
  --mha_dropout 0.2 \
  --mha_heads 4 \
  --lr_scheduler linear \
  --activation_fn swish \
  --num_sanity_val_steps 2 \
  --max_sequence_length 512 \
  --utterance_encoder_type connected_mha \
  --dialogue_encoder_type connected_lstm \
  --pa_encoder_type connected_mha \
  --lstm_num_layers 1 \
  --lstm_bidirectional 1 \
  --fp16 \
  --pretrained_encoder 31_Aug_LM+dialogue+SRL \
  --frozen_layer_names word_embeddings \
done
