# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=4 ../optimize_smooth.py \
--input_model /localssd/hyzhang/llama-2-7b-hf  \
--output_rotation_path /localssd/hyzhang/spinquant/rotation \
--output_dir /localssd/hyzhang/spinquant/output \
--logging_dir /localssd/hyzhang/spinquant/log \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 300 \
--w_bits 4 \
--a_bits 4 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--w_groupsize 32 \
--a_groupsize 32 \
--k_groupsize 128 \
--v_groupsize 128 \
# --a_asym \
# --k_asym \
# --v_asym \
