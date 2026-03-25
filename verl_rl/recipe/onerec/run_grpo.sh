#!/bin/bash
# GRPO Training Script with Two-Stage Rollout
# Two-Stage Rollout: first generate to </think>, then insert <sid_begin> and beam search

set -e

# ============================================================================
# Cluster Configuration (auto-detect from Ray)
# ============================================================================
RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')" 2>/dev/null)

export N_NODES=$(echo $RAY_INFO | awk '{print $1}')
export N_GPUS=$(echo $RAY_INFO | awk '{print $2}')

if [ -z "$N_NODES" ] || [ -z "$N_GPUS" ] || [ "$N_NODES" -eq 0 ]; then
    echo "Could not detect Ray cluster. Using defaults: N_NODES=1, N_GPUS=8"
    export N_NODES=1
    export N_GPUS=8
else
    echo "Detected Ray cluster: $N_NODES nodes, $N_GPUS GPUs per node"
fi

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Model Configuration
# ============================================================================
export BASE_MODEL=${BASE_MODEL:-"/path/to/your/model"}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export TEMPERATURE=${TEMPERATURE:-1}

# ============================================================================
# Batch Size Configuration
# ============================================================================
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-40960}
export TRAIN_BATCH_SIZE=$((N_GPUS * N_NODES))

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-32}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}

# Think mode configuration
export ENABLE_THINK=${ENABLE_THINK:-False}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-False}

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR=${DATA_DIR:-"/path/to/your/data"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_two_stage"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}

# ============================================================================
# Network Configuration (for distributed training)
# ============================================================================
export TCP_NIC=$(ifconfig 2>/dev/null | grep -B1 " "$(hostname -i 2>/dev/null)" " | grep -o "^\w*" || echo "eth0")
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# ============================================================================
# Print Configuration
# ============================================================================
echo "==================================="
echo "GRPO Training with Two-Stage Rollout"
echo "==================================="
echo "Model: $BASE_MODEL"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "Stage2 Beam Size: $STAGE2_BEAM_SIZE"
echo "Enable Think: $ENABLE_THINK"
echo "Enable NonThink: $ENABLE_NONTHINK"
echo "==================================="

# ============================================================================
# Launch Training
# ============================================================================
mkdir -p logs

python3 -u -m recipe.onerec.main_onerec_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=10240 \
    ++data.enable_think=$ENABLE_THINK \
    ++data.enable_nonthink=$ENABLE_NONTHINK \
    ++data.use_force_prefix=$USE_FORCE_PREFIX \
    data.prompt_key='prompt' \
    data.shuffle=True \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py \
    data.custom_cls.name=OneRecDataset \
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=two_stage \
    ++actor_rollout_ref.rollout.backend=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH \
    ++actor_rollout_ref.rollout.stage1_max_tokens=$STAGE1_MAX_TOKENS \
    ++actor_rollout_ref.rollout.stage2_num_tokens=$STAGE2_NUM_TOKENS \
    ++actor_rollout_ref.rollout.stage2_beam_size=$STAGE2_BEAM_SIZE \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpt \
    trainer.total_epochs=20 \
    trainer.val_before_train=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    "$@"
