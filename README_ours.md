# OpenOneRec 代码库详细分析

基于对仓库结构、README 文件和关键脚本的检查，以下是对 OpenOneRec 代码库的详细分析。

## 概览 (Overview)

OpenOneRec 是一个基于大型语言模型（LLMs）的开源生成式推荐框架。它使用 Qwen3 作为骨干（Backbone），并引入了 **Itemic Tokens**（通过量化将物品视为 Token），以弥合推荐 ID 和自然语言之间的差距。

该流程主要包含 5 个主要阶段：

1. **数据准备与分词 (Data Preparation & Tokenization)** (`data/`, `tokenizer/`)，数据主要是纯文本数据（和onerec-think那种一样）以及chat数据（有结构的数据，定义了角色和内容。）
2. **预训练 (Pretraining)** (`pretrain/`)（作用是让模型能够理解推荐数据，但要求也要不遗忘通用知识。具体是类似于onerec-think先更新推荐词表。但又加了一步在推荐数据和通用文本数据的混合数据上进行全参数训练，确保模型在学习推荐模式的同时不会遗忘语言能力。）
3. **监督微调 (Supervised Fine-Tuning, SFT) 与在线蒸馏** (`pretrain/`)（先用chat数据全量微调，以获得回答问题的能力，可以选择是否支持推理，数据也是通用sft+推荐sft。这之后又加了一步蒸馏，让原始的llm来知道当前微调过的llm，再次保证具备通用知识。）
4. **强化学习 (On-policy Distillation & RL)** (`verl_distillation/`, `verl_rl/`)（做了一步GRPO，做了一步强化。数据还是之前的chat数据，但是用rl算法来做全量参数更新。）
5. **评估 (Evaluation)** (`benchmarks/`)

---

## 第一阶段：数据准备与分词 (Phase 1: Data Preparation & Tokenization)

### 1.1 数据准备 (Data Preparation) (`data/`)

此阶段将原始数据集转换为训练流程所需的统一 Parquet 格式。

* **功能 (Function):**
* 从 HuggingFace 下载通用文本和推荐数据集。
* 将原始推荐日志转换为“对话格式”（Chat Format，包含 `messages` 字段，有结构、有角色的对话记录。messages 就是一个列表，里面装着多轮对话的条目。每一条目都有明确的身份（role）和内容（content））或“片段格式”（Segments Format，纯文本）。

    * **输入 (Input):** 原始数据集（JSON/CSV）或 HuggingFace 仓库 ID。

        输入数据主要分为 **元数据 (Metadata)**、**映射表 (Mapping)** 和 **辅助信息 (Auxiliary)**，通常为 Parquet 格式。

        * **(1) 业务元数据 (`onerec_bench_release.parquet`)**
            这是核心输入，包含用户的交互历史、特征等。
            * **关键列:**
                * `uid`: 用户 ID。
                * `history_pid_list`: 历史点击的物品 ID 列表（原始 ID，如 `pid_123`）。
                * `target_pid`: 目标（下一个）点击的物品 ID。
                * `split`: 数据集划分标识（通常 0 代表训练集）。
                * `interaction_type`: 交互行为（如：点赞、关注）。
                * `query`: 搜索关键词（针对交互式推荐任务）。

        * **(2) ID 映射表 (`pid2sid.parquet`)**
            用于将原始物品 ID (PID) 转换为大模型可理解的 Itemic Tokens (SID)。
            * **关键列:**
                * `pid`: 原始物品 ID。
                * `codes`: 一个由 3 个整数组成的数组（如 `[340, 6566, 5603]`），代表残差量化后的 Code。

        * **(3) 辅助信息 (`pid2caption.parquet`)**
            * **关键列:** `pid`, `caption` (物品的文本描述)。
    * **输出 (Output):** 包含 `uuid`、`source`、`messages/segments` 的统一 Parquet 文件。

        输出是统一的 Parquet 格式，根据任务类型分为 **Segments 格式（预训练用）** 和 **Chat 格式（SFT 用）**。

        * **(1) 预训练格式 (Segments Format)**
            * **字段:** `source`, `uuid`, `segments` (JSON 数组), `metadata`。
            * **样例:**
            ```json
            {
                "source": "RecIF_VideoRec_Pretrain",
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "segments": [
                {
                    "type": "text",
                    "text": "用户看过了：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，接着看了：<|sid_begin|><s_a_123><s_b_456><s_c_789><|sid_end|>"
                }
                ],
                "metadata": "{\"uid\": \"user_001\"}"
            }
            ```

        * **(2) SFT 格式 (Chat 格式)**
            * **字段:** `source`, `uuid`, `messages` (对话列表), `metadata`。
            * **样例:**
            ```json
            {
                "source": "RecIF_VideoRec",
                "uuid": "550e8400-e29b-41d4-a716-446655440001",
                "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "你是一个专业的短视频推荐助手。"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "根据我之前的观看历史：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，请推荐下一个视频。"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "我推荐你观看：<|sid_begin|><s_a_111><s_b_222><s_c_333><|sid_end|>"}]
                }
                ],
                "metadata": "{\"uid\": \"user_001\"}"
            }
            ```
* 将数据分片（Shard）为 1000 个样本的块（Chunk）以便高效加载。

    这一步是将通用数据和第 2 步生成的推荐数据合并，并切分成小文件供训练使用。

    * **3.1 `bash prepare_pretrain.sh`**
        * **输入 (Input):**
            1.  **通用预训练数据:** `raw_data/general_text/pretrain/*.parquet`
                * *格式:* Segments 格式。
            2.  **推荐预训练数据:** `output/pretrain_*.parquet` (来自第 2 步)
                * *格式:* Segments 格式。
        * **输出 (Output):**
            * **文件位置:** `output/split_data_pretrain/` 目录。
            * **文件内容:** `part-00000.parquet`, `part-00001.parquet` ...
            * **数据格式:** Segments 格式 (包含 `segments` 列)。
            * **说明:** 这是把上述两种输入数据混合打乱后，每 1000 条存一个文件。还包含一个 `file_list.json` 索引文件。

    * **3.2 `bash prepare_sft.sh`**
        * **输入 (Input):**
            1.  **通用 SFT 数据:** `raw_data/general_text/sft/*.parquet`
                * *格式:* Chat 格式。
            2.  **推荐 SFT 数据:** `output/sft_*.parquet` (来自第 2 步)
                * *格式:* Chat 格式。
        * **输出 (Output):**
            * **文件位置:** `output/split_data_sft/` 目录。
            * **文件内容:** `part-00000.parquet`, `part-00001.parquet` ...
            * **数据格式:** Chat 格式 (包含 `messages` 列)。
            * **说明:** 混合打乱后，每 1000 条存一个文件。

    * **3.3 `bash prepare_distillation.sh`**
        * **输入 (Input):**
            1.  **通用预训练数据:** `raw_data/general_text/pretrain/*.parquet` (注意：这里通常只用通用文本)
                * *格式:* Segments 格式。
        * **输出 (Output):**
            * **文件位置:** `output/onpolicy_distillation.parquet` (单个文件)。
            * **数据格式:** Prompt Only 格式 (通常包含 `prompt` 列，或者仅仅是文本片段)。
            * **说明:** 从海量通用文本中随机抽取了约 20 万条，用于在蒸馏时让模型做“填空题”，以保持通用能力。

    * **3.4 `bash prepare_rl.sh`**
        * **输入 (Input):**
            1.  **推荐 SFT 数据 (特定任务):** 仅读取 `output/` 目录下的以下 5 个文件：
                * `sft_video_rec.parquet`
                * `sft_ad_rec.parquet`
                * `sft_product_rec.parquet`
                * `sft_interactive_rec.parquet`
                * `sft_label_cond_rec.parquet`
                * *格式:* Chat 格式。
        * **输出 (Output):**
            * **文件位置:** `output/rl_data/` 目录 (包含 `train.parquet` 和 `test.parquet`)。
            * **数据格式:** Chat 格式 (包含 `messages` 列)。
            * **说明:** 将这 5 个特定的推荐任务数据合并，并划分了训练集和测试集。这些数据专门用于后续的强化学习训练。

* **执行 (Execution):**

```bash
# 1. 下载数据
# (使用 data/README.md 中的 hf download 命令)

# 2. 处理特定的推荐数据
cd data/onerec_data && bash run.sh

# 3. 分片并为特定阶段做准备
cd data
bash prepare_pretrain.sh     # 用于预训练 (Pretraining)
bash prepare_sft.sh          # 用于监督微调 (SFT)
bash prepare_distillation.sh # 用于蒸馏 (Distillation)
bash prepare_rl.sh           # 用于强化学习 (RL)

```

### 1.2 分词器 (Tokenizer) (`tokenizer/`) （新数据处理）

这是物品 ID 和 LLM Token 之间的桥梁。

* **功能 (Function):** 使用残差 K-Means（Residual K-Means）将连续的物品 Embedding（来自 Qwen3-Embedding 模型）量化为离散的编码（Itemic Tokens）。
* **输入 (Input):** 包含物品 Embedding 的 Parquet 文件。
* **输出 (Output):** 训练好的量化器模型 (`model.pt`) 和物品代码 (`codes.parquet`)。
* **执行 (Execution):**

```bash
cd tokenizer
python train_res_kmeans.py --data_path ./data/embeddings.parquet ...
python infer_res_kmeans.py ... # 为物品生成代码

```

---

## 第二阶段：预训练 (Phase 2: Pretraining) (`pretrain/`)

该模块处理核心的 LLM 训练，使 Qwen3 适应推荐任务。

### 步骤 1：词表扩充 (Vocabulary Expansion)

在训练前，必须将新的 Itemic Tokens 添加到 Qwen3 分词器中。

* **命令:** `bash scripts/expand_qwen3_vocab.sh`
* **输入:** 原始 Qwen3 模型。
* **输出:** 词表扩充后的 Qwen3 模型（后续作为 `base_model_dir` 使用）。如果 ITEMIC_LAYER_N=3 且 VOCAB_SIZE=8192，它就自动生成 s_a_0 到 s_a_8191，s_b_0 到 s_b_8191，s_c_0 到 s_c_8191，再加上 <|sid_begin|> 和 <|sid_end|>。

### 步骤 2：第一阶段预训练（Itemic-Text 对齐）

* **功能:** 冻结 LLM 参数，**仅训练 Itemic Tokens 的 Embedding 层**。这使得新添加的 Itemic Tokens 与预训练的 LLM 空间对齐。
* **输入:** 扩充后的 Qwen3 模型，第一阶段数据集配置 (`examples/dataset_config/stg1.json`)。
* **输出:** 更新了 Embedding 的 Checkpoint。
* **执行:** `bash pretrain/examples/pretrain_stg1.sh` 改成`bash pretrain/run_pretrain_stg1_local.sh`

### 步骤 3：第二阶段预训练（联合预训练 Co-Pretraining）

* **功能:** 在推荐数据和通用文本数据的混合数据上进行**全参数训练**，确保模型在学习推荐模式的同时不会遗忘语言能力。
* **输入:** 第一阶段模型，预训练数据集配置 (`examples/dataset_config/pretrain.json`)。
* **输出:** 预训练好的基座模型 Checkpoint。
* **执行:** `bash examples/pretrain_stg2.sh`  改成`bash pretrain/run_pretrain_stg2_local.

> **注意**: 训练后，使用 `bash scripts/convert_checkpoint_to_hf.sh` 将 Checkpoint 转换为 HuggingFace 格式。

---

## 第三阶段：后训练 (Phase 3: Post-Training) (SFT & Distillation)

### 3.1 监督微调 (SFT) (`pretrain/`)

* **功能:** 指令微调 (Instruction Tuning)，教模型遵循特定的用户命令（例如“推荐一个视频”）。支持“思考模式”（Chain-of-Thought）（如果开启了 "Thinking Mode"，Assistant 的回复里还会包含 `<think>...</think>` 的推理过程，由data/onerec_data/sft/reco_reason.py生成，方式是读取用户的历史标签（比如“动作”、“科幻”），然后套用模板生成文本：“用户历史观看偏好主要集中在动作和科幻类别，因此推荐...”）。全量微调。
* **输入:** 第二阶段模型，SFT 数据集配置 (`examples/dataset_config/sft.json`)。混合了通用 SFT 数据和推荐 SFT 数据。
* **执行:** `bash examples/posttrain_sft.sh`

### 3.2 在线蒸馏 (On-Policy Distillation) (`verl_distillation/`)

* **功能:** 使用教师模型（原始 Qwen3）来指导学生模型（OneRec，也就是经历了Qwen3 -> 词表扩展 -> Stage 1 预训练 -> Stage 2 混合预训练 -> SFT的模型）。作用是通过蒸馏，保留通用推理能力。
* **输入:** 学生模型 (OneRec SFT)，教师模型 (Qwen3)，通用文本 Parquet 数据。
* **输出:** 蒸馏后的模型。
* **执行:**

```bash
cd verl_distillation
export BASE_MODEL=... TEACHER_MODEL=...
bash recipe/onpolicy_distill/run_qwen3_distill.sh

```

---

## 第四阶段：强化学习 (Phase 4: Reinforcement Learning) (`verl_rl/`)

* **功能:** 使用 GRPO（Group Relative Policy Optimization）在 5 个核心推荐任务上进一步将模型与人类偏好对齐。
* **输入:** 蒸馏/SFT 模型，RL 数据集（合并的 train/test parquet）。
* **输出:** 最终经过 RL 对齐的模型。
* **执行:**

```bash
cd verl_rl
bash recipe/onerec/run_grpo.sh

```

---

## 第五阶段：评估 (Phase 5: Evaluation) (`benchmarks/`)

* **功能:** 在 RecIF-Bench 上评估模型，涵盖 8 个任务（如视频推荐、交互式推荐、解释生成等）。
* **输入:** 训练好的模型路径，评估数据 (`./data_v1.0/`)。
* **输出:** 指标日志（Recall@K, AUC 等），位于 `benchmarks/results/`。

```
cd benchmarks
# Set up environment variables
export BENCHMARK_BASE_DIR=$(pwd)
export BENCHMARK_DATA_DIR="../raw_data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"

# Run evaluation (Model Path, Result Name, Enable Thinking)
bash eval_script.sh ../checkpoints/OneRec-1.7B results_1.7B false
```

* **执行:**

```bash
cd benchmarks
bash eval_script.sh <model_path> <result_name> <enable_thinking>

```

这种结构使得 OpenOneRec 能够作为一个完整的工厂，用于构建、训练和评估大型推荐模型 (Large Recommendation Models)。