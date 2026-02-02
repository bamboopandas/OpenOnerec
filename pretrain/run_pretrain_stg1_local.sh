#!/bin/bash

# ================= 配置区域 =================
# 指定使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 模型路径 (刚刚确认过的路径)
MODEL_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/data/code/onerec_pretrain/hf_models/Qwen3-0.6B_itemic"

# 输出路径 (改为您的工作目录)
OUTPUT_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/stg1_opt"

# 数据配置 (注意：请确认此 JSON 里的 sources 路径是否正确)
DATASET_CONFIG="examples/dataset_config/pretrain.json"

# ===========================================

mkdir -p $OUTPUT_DIR
mkdir -p /tmp/_wids_cache

export PYTHONPATH=$PWD:$PYTHONPATH

# Disable proxy for localhost to avoid torch.distributed connection issues
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1

# 设置分布式训练的环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "🚀 开始 Stage 1 预训练..."
echo "📍 模型: $MODEL_DIR"
echo "📍 输出: $OUTPUT_DIR"

# 使用 torchrun 启动
# nproc_per_node = 4 (因为指定了 4 张卡: 2,3,4,5)
# Unset LD_PRELOAD to disable proxychains for torchrun
unset LD_PRELOAD
/home/lkzhang/miniconda3/envs/openonerec/bin/torchrun --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    recipes/train_qwen3.py \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config $DATASET_CONFIG \
    --freeze_llm \
    --use_tie_weights \
    --start_optimize_embedding_index 151669 \
    --model_class Qwen3ForCausalLM \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length 32768 \
    --learning_rate 2e-4 \
    --min_lr 1e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 200 \
    --num_training_steps 2000 \
    --save_checkpoint_per_step 50 \
    --minibatch_size 4096 \
    --logging_per_step 5 \
    --use_fp32_weight \
    --seed 19260817 \
    --enable_gradient_checkpointing \
    --use_chunked_loss_computer
