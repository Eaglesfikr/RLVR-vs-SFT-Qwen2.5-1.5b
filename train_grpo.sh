#!/bin/bash
# GRPO training for Qwen2.5-1.5B on GSM8K
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$SCRIPT_DIR/data/train.parquet" \
    +model.override_config.attn_implementation=eager \
    data.val_files="$SCRIPT_DIR/data/gsm8k_test.parquet" \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    data.max_prompt_length=512 \
    data.max_response_length=3584 \
    data.truncation=left \
    actor_rollout_ref.model.path="$SCRIPT_DIR/models/qwen2-5-1-5b-instruct" \
    +actor_rollout_ref.model.local_tokenizer_path="$SCRIPT_DIR/models/qwen2-5-1-5b-instruct" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.max_position_embeddings=4096 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enforce_eager=true \
    reward.custom_reward_function.path="$SCRIPT_DIR/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    critic.enable=false \
    trainer.total_epochs=13 \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.val_before_train=false \
    trainer.default_local_dir="$SCRIPT_DIR/outputs/grpo_$(date +%d-%m_%H%M)" \
    trainer.project_name=verl-gsm8k-grpo-train \
    trainer.experiment_name=qwen2.5-1.5b-instruct-grpo \
    +actor_rollout_ref.actor.checkpoint.save_contents='[hf_model]' \
    trainer.resume_mode=disable \
    "$@"
