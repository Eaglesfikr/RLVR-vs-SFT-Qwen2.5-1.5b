#!/bin/bash
# SFT training on GSM8K train set
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files="$SCRIPT_DIR/data/sft_train.parquet" \
    +model.override_config.attn_implementation=eager \
    data.train_batch_size=4 \
    data.micro_batch_size_per_gpu=1 \
    data.use_dynamic_bsz=false \
    data.max_length=4096 \
    data.truncation=left \
    model.path=/root/autodl-tmp/RLVR-vs-SFT-Qwen2.5-1.5b/models/Qwen2.5-1.5B-Instruct \
    model.enable_gradient_checkpointing=true \
    +model.override_config.max_position_embeddings=4096 \
    engine.use_torch_compile=true \
    optim.lr=5e-7 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.05 \
    checkpoint.save_contents='[hf_model]' \
    trainer.total_epochs=7 \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=2000 \
    trainer.default_local_dir="$SCRIPT_DIR/outputs/sft_$(date +%d-%m_%H%M)" \
    trainer.project_name=verl-gsm8k-sft \
    trainer.experiment_name=qwen2.5-1.5b-instruct-sft-gsm8k-train \
    trainer.resume_mode=disable \
    "$@"
