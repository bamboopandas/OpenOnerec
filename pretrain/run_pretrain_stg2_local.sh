#!/bin/bash

# ================= 配置区域 =================
# 指定使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=2,3,4,5

# Stage 1 输出的 Converted 模型路径
# 注意：运行此脚本前，请确保已运行 checkpoint conversion 脚本将 Stage 1 结果转换为 HF 格式
# 例如：bash scripts/convert_checkpoint_to_hf.sh <BASE_MODEL_PATH> <STAGE1_OUTPUT_DIR> <STEP>
STAGE1_OUTPUT_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/stg1_opt"
# 假设 Stage 1 跑了 2000 步并已转换
MODEL_DIR="${STAGE1_OUTPUT_DIR}/step2000/global_step2000/converted"

# 输出路径
OUTPUT_DIR="/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/output/stg2_opt"

# 数据配置
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
export MASTER_PORT=29501 # 使用不同的端口以防冲突

echo "🚀 开始 Stage 2 预训练..."
echo "📍 加载 Stage 1 模型: $MODEL_DIR"
echo "📍 输出: $OUTPUT_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 错误: 找不到模型目录 $MODEL_DIR"
    echo "请检查路径，或确认是否已运行 checkpoint 转换脚本。"
    exit 1
fi

# Unset LD_PRELOAD to disable proxychains for torchrun
unset LD_PRELOAD

# 使用 torchrun 启动
# nproc_per_node = 4 (对应上面 CUDA_VISIBLE_DEVICES 的数量)
/home/lkzhang/miniconda3/envs/openonerec/bin/torchrun --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    recipes/train_qwen3.py \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config $DATASET_CONFIG \
    --use_tie_weights \
    --model_class Qwen3ForCausalLM \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length 32768 \
    --learning_rate 2e-4 \
    --min_lr 1e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --num_training_steps 5000 \
    --save_checkpoint_per_step 50 \
    --minibatch_size 4096 \
    --logging_per_step 5 \
    --use_fp32_weight \
    --seed 19260817 \
    --enable_gradient_checkpointing \
    --use_chunked_loss_computer
