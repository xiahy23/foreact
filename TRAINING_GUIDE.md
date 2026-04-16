# ForeAct 训练与推理详细指南

本文档是 `dev` 分支上所有训练、推理、AutoML 脚本的完整使用说明。

---

## 目录

- [1. 环境配置](#1-环境配置)
- [2. 数据准备](#2-数据准备)
- [3. 微调训练](#3-微调训练)
- [4. AutoML 超参搜索](#4-automl-超参搜索)
- [5. 推理](#5-推理)
- [6. 工具脚本](#6-工具脚本)
- [7. 配置文件详解](#7-配置文件详解)
- [8. 分布式训练说明](#8-分布式训练说明)

---

## 1. 环境配置

```bash
# 创建 conda 环境并安装依赖
bash environment_setup.sh foreact

# 或手动：
conda create -n foreact python=3.10 -y
conda activate foreact
conda install ffmpeg=7.1.1 -c conda-forge
pip install -r requirements.txt
```

**关键依赖**：PyTorch, transformers, diffusers, accelerate, deepspeed, optuna, gradio, volcenginesdkarkruntime (VLM API)

**环境变量**（根据需要设置）：

| 变量 | 说明 | 使用场景 |
|------|------|----------|
| `VOLCENKEY` | 火山引擎 API Key | AutoML VLM 评分、Doubao 推理 |
| `WANDB_PROJECT` | W&B 项目名（默认 `VisualForesight`）| 训练日志 |
| `FORCE_VIDEO_BACKEND` | 视频解码后端（`pyav` 推荐）| 数据加载 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU | 所有训练/推理 |

---

## 2. 数据准备

将数据放置于 `./datasets/` 下，目录结构如下：

```
datasets/
├── bridge_orig_lerobot/          # 主训练数据集
│   ├── videos/
│   │   └── chunk-000/
│   │       └── observation.images.image_0/
│   │           ├── episode_000000.mp4
│   │           ├── episode_000001.mp4
│   │           └── ...
│   └── subtask_dataset/          # 子任务数据集
├── all_eval/                     # AllEval 评估数据集（可选）
├── top10000_episodes.json        # 过滤后的 episode 列表
└── cot_by_episode.json           # Chain-of-Thought 子任务标注
```

**预训练权重**：从 [HuggingFace](https://huggingface.co/mit-han-lab/foreact-pretrained) 下载并放至 `./foreact-pretrained/`。

---

## 3. 微调训练

### 3.1 基础微调 (bridge)

**配置文件**：`configs/finetune.yaml`

```bash
# 方式一：使用封装脚本（自动处理 torchrun 参数）
bash scripts/run_finetune.sh

# 方式二：手动执行
torchrun --nproc_per_node=8 train.py \
    --config_file finetune.yaml \
    --run_name my_bridge_finetune
```

**核心训练参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `per_device_train_batch_size` | 16 | 每 GPU 的 batch size |
| `learning_rate` | 1e-5 | 学习率 |
| `num_train_epochs` | 5.0 | 训练轮数 |
| `lr_scheduler_type` | `constant_with_warmup` | 学习率调度器 |
| `warmup_steps` | 0 | 预热步数 |
| `save_steps` | 1000 | Checkpoint 保存间隔 |
| `deepspeed` | `configs/zero1.json` | DeepSpeed ZeRO-1 |
| `bf16` | true | BF16 混合精度 |

**数据参数**：

| 参数 | 说明 |
|------|------|
| `data_path` | 数据集根目录 (`./datasets`) |
| `camera_key` | 视频帧的 key (`observation.images.image_0`) |
| `target_image_size` | 目标图像尺寸 `[480, 640]` |
| `filtered_episodes_path` | Episode 过滤列表（JSON） |
| `cot_json_path` | CoT 标注文件 |
| `subtask_data_path` | 子任务数据集路径 |
| `target_frame_offset` | 目标帧偏移量（默认不使用） |

**输出**：`./checkpoints/finetuned_bridge_4/checkpoint-{step}/`

---

### 3.2 Offset 模式微调

**配置文件**：`configs/finetune_offset6.yaml`

此配置使用 **固定帧偏移** 策略（offset=6）：模型预测当前帧后第 6 帧的观测。

```bash
torchrun --nproc_per_node=8 train.py \
    --config_file finetune_offset6.yaml \
    --run_name finetune_offset6
```

**与基础微调的区别**：
- `target_frame_offset: 6` — 预测未来第 6 帧
- 不使用 CoT / 子任务数据 / Episode 过滤
- 适合需要自定义数据路径的场景（通过 `custom_data_path` 指定）

**输出**：`./checkpoints/finetuned_offset6/`

---

### 3.3 AllEval + Bridge 混合数据微调

**配置文件**：`configs/finetune_alleval_bridge.yaml`

```bash
# 推荐方式（自动处理 OOM、自动降低 batch size 重试）
bash scripts/run_finetune_alleval_bridge.sh

# 手动方式
accelerate launch --num_processes 8 --mixed_precision bf16 \
    train.py \
    --config_file finetune_alleval_bridge.yaml \
    --run_name finetune_alleval_bridge_offset6
```

**特殊配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `custom_data_path` | `./datasets/all_eval` | AllEval 数据集路径 |
| `balance_datasets` | true | 对原始和新数据集 1:1 平衡采样 |
| `_gradient_checkpointing` | true | 启用梯度检查点（省显存） |
| `lr_scheduler_type` | `cosine_with_min_lr` | 余弦退火调度器 |
| `lr_scheduler_kwargs.min_lr` | 1e-6 | 最小学习率 |
| `gradient_accumulation_steps` | 2 | 梯度累积步数 |
| `save_total_limit` | 3 | 最多保留 3 个 checkpoint |

**run_finetune_alleval_bridge.sh 特性**：
- GPU 显存检查：等待所有 GPU 有 ≥70GB 可用显存才启动
- OOM 自动重试：依次尝试 batch_size 4→8→16→32→64
- 环境预设：`FORCE_VIDEO_BACKEND=pyav`

**输出**：`./checkpoints/` + W&B 日志

---

## 4. AutoML 超参搜索

### 4.1 Optuna AutoML (pure_orig / mixed)

**脚本**：`automl_optuna.py` + `run_automl.sh`

Optuna 采样超参 → 生成 trial YAML → torchrun 训练 → VLM 自动打分 → 反馈 Optuna 闭环。

```bash
# 标准启动（pure_orig 模式）
bash run_automl.sh

# 或自定义参数
python automl_optuna.py \
    --vlm_model doubao-seed-2-0-lite-260215 \
    --n_trials 200 \
    --num_epochs 5 \
    --gpus 8 \
    --data_mode pure_orig   # 或 mixed
```

**参数详解**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vlm_model` | `doubao-seed-2-0-lite-260215` | VLM 评分模型 |
| `--n_trials` | 200 | 总试验次数 |
| `--num_epochs` | 5 | 每 trial 训练轮数 |
| `--gpus` | 8 | GPU 数量 |
| `--data_mode` | `pure_orig` | `pure_orig`=纯原始数据+CoT / `mixed`=过滤原始+子任务 |
| `--dry_run` | false | 仅打印命令不实际运行 |
| `--master_port` | 25010 | torchrun 端口 |

**搜索空间**：
- Learning rate: `[1e-5, 7e-5]` (对数均匀)
- Batch size: `{4, 8}`
- LR Scheduler: `{linear, constant_with_warmup, cosine_with_min_lr}`
- Warmup steps: `[0, 150]`

**VLM 评分维度**（每维 0-20 分，共 100 分）：

| 维度 | 含义 |
|------|------|
| `scene_consistency` | 场景一致性 |
| `gripper_clarity` | 抓取器清晰度 |
| `physics_plausibility` | 物理合理性 |
| `task_progress` | 任务进展 |
| `no_artifacts` | 无伪影 |

**产出**：
- Trial 配置：`configs/automl_trials/trial_NNN.yaml`
- Checkpoint：`checkpoints/automl/trial_NNN/`
- 评分结果：`batch_results_automl/trial_NNN/vlm_scores.json`
- 日志：`automl_run.log` 或 `automl_pureorig_run.log`

---

### 4.2 AllEval AutoML

**脚本**：`automl_alleval.py` + `scripts/run_automl_alleval.sh`

基于 5-epoch 基线 checkpoint (`checkpoint-17690`) 继续搜索，10 epoch 训练。

```bash
# 标准启动
bash scripts/run_automl_alleval.sh

# 或手动
python automl_alleval.py \
    --vlm_model doubao-seed-2-0-lite-260215 \
    --n_trials 12 \
    --num_epochs 10 \
    --gpus 8
```

**扩展搜索空间**（相比 4.1）：

| 参数 | 范围 |
|------|------|
| `learning_rate` | `[3e-6, 3e-5]` |
| `lr_scheduler_type` | `{cosine_with_min_lr, cosine, linear, constant_with_warmup}` |
| `warmup_steps` | `[50, 500]` |
| `batch_size` | `{4, 8}` |
| `gradient_accumulation_steps` | `{2, 4}` |
| `unfreeze_mllm` | `{True, False}` — 是否解冻 MLLM 骨干网络 |

**产出**：
- Trial 配置：`configs/automl_alleval_trials/trial_NNN.yaml`
- Checkpoint：`checkpoints/automl_alleval/trial_NNN/`
- 评分结果：`batch_results_automl_alleval/trial_NNN/vlm_scores.json`
- 日志：`automl_alleval.log`

---

### 4.3 恢复中断的 Trial

```bash
# 恢复 trial_001（需先手动创建 trial_001_resume.yaml）
bash resume_trial.sh 001

# 可选参数
bash resume_trial.sh 001 8    # 8 GPU
bash resume_trial.sh 001 8 25020  # 指定 master_port
```

该脚本执行：
```bash
torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    train.py --config_file configs/automl_trials/trial_${TRIAL_ID}_resume.yaml
```

日志输出到 `resume_trial_{TRIAL_ID}.log`。

---

### 4.4 补评缺失分数

当 AutoML 运行中断导致部分 trial 缺少 VLM 评分时：

```bash
python rescore_missing.py \
    --vlm_model doubao-seed-2-0-lite-260215 \
    --results_dir ./batch_results_automl \
    --batch_size 20

# 预览模式（不实际调用 API）
python rescore_missing.py --vlm_model ... --dry_run
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vlm_model` | (必填) | VLM 评分模型 |
| `--results_dir` | `./batch_results_automl` | trial 结果目录 |
| `--batch_size` | 20 | 每批评分的图片数 |
| `--cot_path` | (可选) | CoT 标注文件 |
| `--dry_run` | false | 仅列出需补评的 trial |

---

## 5. 推理

### 5.1 CLI 单图推理

```bash
python app_cli.py \
    --checkpoint_path ./checkpoints/finetuned_bridge_4 \
    --prompt "pick up the red block and place it on the blue plate" \
    --input_image ./test/frame.png \
    --output_dir ./results \
    --guidance_scale 4.5 \
    --image_guidance_scale 1.5 \
    --num_inference_steps 8 \
    --seed 42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_path` | (必填) | 模型 checkpoint 路径 |
| `--prompt` | (必填) | 任务描述文本 |
| `--input_image` | (可选) | 输入图像路径（启用图像引导） |
| `--guidance_scale` | 4.5 | 文本引导强度 |
| `--image_guidance_scale` | 1.5 | 图像引导强度 |
| `--num_inference_steps` | 8 | 扩散步数 |
| `--seed` | random | 随机种子 |
| `--num_images_per_prompt` | 1 | 每 prompt 生成图片数 |

**输出**：`{output_dir}/generated_{timestamp}_seed{seed}_{idx}.png`

---

### 5.2 Gradio Web UI

```bash
python app.py --checkpoint_path ./checkpoints/finetuned_bridge_4
```

访问 `http://localhost:7860`，提供交互式界面：
- 文本输入框（prompt）
- 图像上传（输入当前观察）
- Guidance scale 滑块 (1-30)
- Image guidance scale 滑块 (1-30)
- Seed 输入框
- 实时预测未来帧

---

### 5.3 批量视频推理 (batch_inference)

逐视频提取帧 → 预测未来帧 → 生成 HTML 对比报告。

```bash
python batch_inference.py \
    --checkpoint_path ./checkpoints/finetuned_bridge_4 \
    --test_dir ./test \
    --output_dir ./batch_results \
    --frame_interval 6 \
    --max_frames_per_video 8 \
    --guidance_scale 4.5 \
    --image_guidance_scale 1.5 \
    --num_inference_steps 8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_path` | (必填) | Checkpoint 路径 |
| `--test_dir` | (必填) | 测试视频目录 |
| `--output_dir` | `./batch_results` | 输出目录 |
| `--frame_interval` | 6 | 帧提取间隔 |
| `--max_frames_per_video` | 8 | 每视频最大处理帧数 |
| `--num_inference_steps` | 8 | 扩散步数 |

**输出**：
```
batch_results/
├── video_name_1/
│   ├── frame_0_input.png
│   ├── frame_0_predicted.png
│   ├── frame_1_input.png
│   ├── frame_1_predicted.png
│   └── ...
├── video_name_2/
└── index.html          # 总览HTML（含并排对比）
```

---

### 5.4 全帧推理 (infer_all_frames)

对整个数据集的所有视频逐帧生成预测，支持多 GPU 多 Worker 并行。

```bash
# 推荐方式：使用启动脚本（8 GPU × 3 workers = 24 并行）
bash run_infer_all.sh

# 手动单 worker
CUDA_VISIBLE_DEVICES=0 python infer_all_frames.py \
    --checkpoint_path ./checkpoints/automl_pureorig/trial_000/checkpoint-best \
    --video_dir ./datasets/bridge_orig_lerobot/videos \
    --cot_path ./datasets/cot_by_episode.json \
    --worker_id 0 \
    --num_workers 1 \
    --guidance_scale 4.5 \
    --num_inference_steps 8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_path` | (必填) | Checkpoint 路径 |
| `--video_dir` | `./datasets/bridge_orig_lerobot/videos` | 视频目录 |
| `--cot_path` | `./datasets/cot_by_episode.json` | CoT 标注（动态 prompt） |
| `--worker_id` | (必填) | Worker ID |
| `--num_workers` | (必填) | 总 Worker 数 |
| `--num_inference_steps` | 8 | 扩散步数 |

**run_infer_all.sh 配置**：
```bash
NUM_GPUS=8         # GPU 数量
PROCS_PER_GPU=3    # 每 GPU 进程数（共 24 workers）
CHECKPOINT="./checkpoints/automl_pureorig/trial_000/..."
```

**任务分配**：Worker `i` 处理满足 `episode_id % num_workers == worker_id` 的 episode。

**输出**：`datasets/bridge_orig_lerobot/videos/chunk-*/observation.images.image_0_foreact/episode_*.mp4`

---

### 5.5 Doubao VLM 评估基线

使用火山引擎 Doubao Seedream 5.0 API 生成预测，作为评估基线对比。

```bash
export VOLCENKEY="your-api-key"
python refer_to_doubao.py \
    --test_dir ./test \
    --output_dir ./batch_results_doubao
```

**输出**：与 batch_inference 相同格式的 HTML 报告。

---

## 6. 工具脚本

### 6.1 视频三联拼接

将 3 个视频源（如原始、ForeAct 预测、其他方法预测）水平拼接为对比视频。

```bash
python merge_triplet_videos.py \
    --dir_a ./datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0 \
    --dir_b ./datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0_foreact \
    --dir_c ./datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0_other \
    --output_dir ./merged_videos \
    --max_videos 10
```

| 参数 | 说明 |
|------|------|
| `--dir_a/b/c` | 三个视频目录（文件名需一一对应） |
| `--output_dir` | 输出目录 |
| `--max_videos` | 最多处理视频数（0=全部） |

---

### 6.2 数据集验证

```bash
# 验证数据集加载是否正确
python verify_datasets.py
# 输出样本图到 verify_output/

# 调试数据集 pair 关系
python debug_dataset.py
# 输出可视化到 debug_dataset_pairs.png
```

---

## 7. 配置文件详解

### 7.1 YAML 配置字段参考

所有 YAML 配置文件的完整字段说明（以 `configs/finetune.yaml` 为主参考）：

#### 模型参数

```yaml
mllm_id: "google/gemma-2-2b-it"                                  # MLLM 模型 ID
diffusion_model_id: "Efficient-Large-Model/Sana_1600M_512px_diffusers"  # 扩散模型 ID
vae_id: ""                       # VAE 模型（空=使用扩散模型自带）
scheduler_id: ""                 # Scheduler（空=使用扩散模型自带）
system_prompt: ""                # 系统提示词
```

#### 数据参数

```yaml
data_path: "./datasets"                                    # 数据集根目录
camera_key: "observation.images.image_0"                   # 摄像头 key
target_image_size: [480, 640]                              # [H, W] 目标分辨率
filtered_episodes_path: "datasets/top10000_episodes.json"  # Episode 过滤列表
cot_json_path: "datasets/cot_by_episode.json"              # CoT 标注
subtask_data_path: "datasets/bridge_orig_lerobot/subtask_dataset"
custom_data_path: ""                                       # 自定义数据路径
target_frame_offset: 0                                     # 目标帧偏移（0=随机）
balance_datasets: false                                    # 是否平衡多数据集
```

#### 训练参数

```yaml
per_device_train_batch_size: 16         # 每 GPU batch size
gradient_accumulation_steps: 1          # 梯度累积
learning_rate: 1.0e-5                   # 学习率
lr_scheduler_type: "constant_with_warmup"  # 调度器类型
lr_scheduler_kwargs:                    # 调度器额外参数
  min_lr: 1.0e-6                        #   (cosine_with_min_lr 时使用)
warmup_steps: 0                         # 预热步数
num_train_epochs: 5.0                   # 训练轮数
save_steps: 1000                        # 保存间隔
save_total_limit: null                  # 最多保留 checkpoint 数（null=不限）
bf16: true                              # BF16 混合精度
seed: 42                                # 随机种子
dataloader_num_workers: 4               # DataLoader 进程数
deepspeed: "configs/zero1.json"         # DeepSpeed 配置
output_dir: "./checkpoints/xxx"         # 输出目录
run_name: "my_run"                      # W&B run 名称
report_to: "wandb"                      # 日志上报方式
```

#### 冻结参数与梯度检查点

```yaml
freeze_modules: ["vae", "mllm_backbone"]   # 冻结的模块列表
_gradient_checkpointing: false              # 梯度检查点（省显存但更慢）
resume_from_checkpoint: "./foreact-pretrained"  # 恢复训练的 checkpoint
```

**可用的 lr_scheduler_type 选项**：
- `constant_with_warmup` — 预热后恒定学习率
- `linear` — 线性衰减
- `cosine` — 余弦退火
- `cosine_with_min_lr` — 有最小学习率的余弦退火（需配合 `lr_scheduler_kwargs.min_lr`）

---

### 7.2 DeepSpeed 配置

**文件**：`configs/zero1.json`

```json
{
  "fp16": { "enabled": "auto" },
  "bf16": { "enabled": "auto" },
  "zero_optimization": { "stage": 1 },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 0.5,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "steps_per_print": 100
}
```

使用 ZeRO Stage 1（优化器状态分片），`"auto"` 值由 HuggingFace Trainer 自动设置。

---

## 8. 分布式训练说明

### torchrun 模式（基础微调 + AutoML）

```bash
torchrun --nproc_per_node=8 --master_port=25010 train.py --config_file finetune.yaml
```

- 通过 `scripts/setup.sh` 自动检测 SLURM 环境
- 支持多节点：`--nnodes`, `--node_rank`, `--master_addr`

### accelerate 模式（AllEval 微调）

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 train.py --config_file ...
```

- `scripts/run_finetune_alleval_bridge.sh` 内置 OOM 自动重试逻辑
- 按 batch_size `{4, 8, 16, 32, 64}` 依次降级尝试

### SLURM 支持

`scripts/setup.sh` 自动检测以下 SLURM 变量：

| SLURM 变量 | 用途 | 默认值 |
|------------|------|--------|
| `SLURM_JOB_NUM_NODES` | 节点数 | 1 |
| `SLURM_GPUS_ON_NODE` | 每节点 GPU 数 | 4 |
| `SLURM_NODEID` | 节点 rank | 0 |

若在非 SLURM 环境，默认使用 `127.0.0.1:25002`，4 GPU。

---

## 快速参考

| 任务 | 命令 |
|------|------|
| 基础微调 | `bash scripts/run_finetune.sh` |
| AllEval 混合微调 | `bash scripts/run_finetune_alleval_bridge.sh` |
| AutoML 超参搜索 | `bash run_automl.sh` |
| AllEval AutoML | `bash scripts/run_automl_alleval.sh` |
| 恢复中断 Trial | `bash resume_trial.sh 001` |
| 补评 VLM 分数 | `python rescore_missing.py --vlm_model ...` |
| CLI 推理 | `python app_cli.py --checkpoint_path ... --prompt ...` |
| Web UI | `python app.py --checkpoint_path ...` |
| 批量推理报告 | `python batch_inference.py --checkpoint_path ... --test_dir ...` |
| 全帧并行推理 | `bash run_infer_all.sh` |
| 视频三联拼接 | `python merge_triplet_videos.py --dir_a ... --dir_b ... --dir_c ... --output_dir ...` |
| 数据集验证 | `python verify_datasets.py` |
