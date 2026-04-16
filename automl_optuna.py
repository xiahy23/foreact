#!/usr/bin/env python3
"""
Agentic AutoML：Optuna + 豆包 VLM 自动调参闭环系统
===================================================
用于 ForeAct 视觉预测模型的全自动超参数优化。

闭环流程：
  1. Optuna 采样超参数
  2. 自动生成 trial 配置 → 拉起 torchrun 训练
  3. 训练结束后调用 batch_inference.py 生成预测图像
  4. 豆包多模态 VLM 对图像质量打分
  5. 分数回传 Optuna，引导下一轮搜索

用法：
  # 先设置 API Key
  export VOLCENKEY="your-api-key"

  # 完整运行（需指定豆包视觉模型 endpoint ID）
  python automl_optuna.py --vlm_model ep-XXXXXXXX --n_trials 20

  # 干跑测试（验证各模块，不实际训练）
  python automl_optuna.py --vlm_model ep-XXXXXXXX --dry_run

  export VOLCENKEY="8dc53ee3-fddc-47b5-b419-5888d5e057ff" && nohup python3 automl_optuna.py --vlm_model doubao-seed-2-0-lite-260215 --n_trials 200 --num_epochs 5 --gpus 8 > automl_run.log 2>&1 &
"""
import argparse
import base64
import copy
import glob
import io
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("automl_optuna.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("automl")

# ============================================================
# 常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "finetune.yaml"
DEFAULT_TEST_DIR = PROJECT_ROOT / "test"

# 测试视频名 → 任务 prompt（和 batch_inference.py 一致）
VIDEO_PROMPTS = {
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_False_src_on_target_False":
        "stack green cube on yellow cube",
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False":
        "put eggplant in basket",
    "failure_obj_episode_3_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False":
        "put spoon on cloth",
    "failure_obj_episode_45_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_False_consecutive_grasp_False_src_on_target_False":
        "put carrot on plate",
    "success_obj_episode_1_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put eggplant in basket",
    "success_obj_episode_36_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put carrot on plate",
    "success_obj_episode_43_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put spoon on cloth",
    "success_obj_episode_60_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "stack green cube on yellow cube",
}


# ============================================================
# 模块一: VLM 视觉裁判 (Vision Evaluator)
# ============================================================
class VLMEvaluator:
    """使用豆包多模态大模型 API 对 (输入图, 预测图) 进行质量评估。

    评估维度（各 0-20 分，满分 100）：
      1. scene_consistency  — 场景一致性（背景/桌面/无关物体不变）
      2. gripper_clarity     — 夹爪清晰度（形态完整、无模糊变形）
      3. physics_plausibility— 物理合理性（无隔空取物、无穿模）
      4. task_progress       — 任务进展合理性（朝目标做出微小但正确的动作）
      5. no_artifacts        — 无杂物/无偷懒（无多余物体生成、没有直接复制输入）
    """

    SCORING_PROMPT = (
        '你是一个专业的机器人视觉预测模型评审专家。\n'
        '我会给你两张图片：\n'
        '- 第一张是"输入图"（机械臂操作场景的当前状态）\n'
        '- 第二张是"预测图"（模型预测的下一步状态）\n'
        '- 当前任务是："{task_prompt}"\n\n'
        '请你严格按照以下 5 个维度对预测图进行打分（每个维度 0-20 分）：\n\n'
        '1. **scene_consistency**（场景一致性，0-20）：\n'
        '   - 背景、桌面、光照是否和输入图一致\n'
        '   - 未被操作的物体是否保持不变\n'
        '   - 扣分项：出现了输入图中不存在的物体、背景发生变化\n\n'
        '2. **gripper_clarity**（夹爪清晰度，0-20）：\n'
        '   - 机械臂夹爪是否清晰可辨、结构完整\n'
        '   - 夹爪是否保持了正确的金属双叉形态\n'
        '   - 扣分项：夹爪模糊、变形、变成人手、出现多余部件\n'
        '   - 注意：夹爪可以绕z轴旋转，此时看上去不是标准的双叉形态，但只要没有模糊变形、没有多余部件，仍然可以得满分\n\n'
        '3. **physics_plausibility**（物理合理性，0-20）：\n'
        '   - 物体运动是否符合物理规律（有接触才能移动）\n'
        '   - 是否存在"隔空取物"（物体没被夹住却悬浮移动）\n'
        '   - 夹爪与物体是否存在穿模（相互穿过）\n'
        '   - 扣分项：物体凭空悬浮、穿模、超前移动\n\n'
        '   - 典型扣分例子：输入图胡萝卜横放，且夹爪未对准胡萝卜，预测图胡萝卜竖放，夹爪对准胡萝卜中部，即胡萝卜没有接触夹爪却移动了，这种情况就直接0分，因为存在隔空取物\n\n'
        '4. **task_progress**（任务进展合理性，0-20）：\n'
        '   - 预测图是否展示了朝着任务目标的合理微小进展\n'
        '   - 夹爪朝向和物体对齐情况是否合理\n'
        '   - 扣分项：完全没有进展、进展方向错误、夹取方向不对\n\n'
        '5. **no_artifacts**（无杂物/无偷懒，0-20）：\n'
        '   - 是否存在凭空生成的杂物、噪点、异常色块\n'
        '   - 是否直接复制了输入图（"偷懒"模式，两张图几乎一模一样）\n'
        '   - 扣分项：有明显杂物生成、完全复制输入图\n\n'
        '请严格以如下 JSON 格式返回，不要输出任何其他内容：\n'
        '```json\n'
        '{\n'
        '    "scene_consistency": <0-20>,\n'
        '    "gripper_clarity": <0-20>,\n'
        '    "physics_plausibility": <0-20>,\n'
        '    "task_progress": <0-20>,\n'
        '    "no_artifacts": <0-20>,\n'
        '    "reasoning": "<一句话简短说明主要问题>"\n'
        '}\n'
        '```'
    )

    def __init__(self, api_key: str, model_id: str):
        from volcenginesdkarkruntime import Ark

        self.client = Ark(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.model_id = model_id
        logger.info(f"VLM Evaluator 初始化完成，模型: {model_id}")

    # ---- 工具方法 ----

    @staticmethod
    def _pil_to_base64_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    def _parse_scores(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """从 VLM 回复中解析 JSON 评分"""
        # 优先匹配 ```json ... ``` 块
        m = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        text = m.group(1) if m else None
        if text is None:
            m = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            text = m.group(0) if m else None
        if text is None:
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        required = [
            "scene_consistency", "gripper_clarity",
            "physics_plausibility", "task_progress", "no_artifacts",
        ]
        for k in required:
            if k not in data:
                return None
            v = data[k]
            if not isinstance(v, (int, float)):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = 0
            data[k] = max(0, min(20, v))

        data["total_score"] = sum(data[k] for k in required)
        data.setdefault("reasoning", "")
        return data

    @staticmethod
    def _fail_score(reason: str) -> Dict[str, Any]:
        return {
            "scene_consistency": 0, "gripper_clarity": 0,
            "physics_plausibility": 0, "task_progress": 0,
            "no_artifacts": 0, "total_score": 0,
            "reasoning": reason,
        }

    # ---- 核心评分 ----

    def score_image_pair(
        self,
        input_img: Image.Image,
        pred_img: Image.Image,
        task_prompt: str,
        retry: int = 3,
    ) -> Dict[str, Any]:
        """对单组 (输入图, 预测图) 进行 VLM 评分"""
        input_uri = self._pil_to_base64_uri(input_img)
        pred_uri = self._pil_to_base64_uri(pred_img)
        prompt = self.SCORING_PROMPT.replace("{task_prompt}", task_prompt)

        for attempt in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": input_uri}},
                            {"type": "image_url", "image_url": {"url": pred_uri}},
                        ],
                    }],
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()
                scores = self._parse_scores(raw)
                if scores:
                    return scores
                logger.warning(f"VLM 返回非 JSON（第{attempt+1}次）: {raw[:200]}")
            except Exception as e:
                logger.warning(f"VLM API 调用失败（第{attempt+1}次）: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)

        return self._fail_score("所有 API 调用均失败")

    def score_trial_output(self, output_dir: str, bottom_k: int = 10, max_pairs: int = 0) -> float:
        """对一个 trial 的所有输出图像评分。

        评分策略：对全部图像对打分，取分数最低的 bottom_k 张的平均值。
        这样能更好地反映模型的"最差表现"，避免被高分掩盖缺陷。

        Args:
            output_dir: batch_inference.py 的输出目录
            bottom_k:   取最低 k 个分数求平均（默认 10）
            max_pairs:  最多评估多少组（0=全部，>0 则均匀采样，用于 dry_run 快速验证）
        """
        pairs: List[Tuple[str, str, str]] = []
        for video_name, task_prompt in VIDEO_PROMPTS.items():
            video_dir = os.path.join(output_dir, video_name)
            if not os.path.isdir(video_dir):
                continue
            for inp in sorted(glob.glob(os.path.join(video_dir, "*_input.png"))):
                pred = inp.replace("_input.png", "_predicted.png")
                if os.path.isfile(pred):
                    pairs.append((inp, pred, task_prompt))

        if not pairs:
            logger.error(f"在 {output_dir} 中未找到图像对")
            return 0.0

        # 可选：均匀采样控制 API 开销（dry_run 时使用）
        if max_pairs > 0 and len(pairs) > max_pairs:
            random.seed(42)
            pairs = random.sample(pairs, max_pairs)

        logger.info(f"对 {len(pairs)} 组图像对进行 VLM 评分...")
        all_scores: List[float] = []
        detail_log: List[Dict] = []

        for i, (inp_path, pred_path, task_prompt) in enumerate(pairs):
            try:
                input_img = Image.open(inp_path).convert("RGB")
                pred_img = Image.open(pred_path).convert("RGB")
            except Exception as e:
                logger.warning(f"读取图像失败: {e}")
                continue

            scores = self.score_image_pair(input_img, pred_img, task_prompt)
            all_scores.append(scores["total_score"])
            detail_log.append({**scores, "input_path": inp_path, "pred_path": pred_path})
            logger.info(
                f"  [{i+1}/{len(pairs)}] total={scores['total_score']:.0f} | "
                f"scene={scores['scene_consistency']:.0f} grip={scores['gripper_clarity']:.0f} "
                f"phys={scores['physics_plausibility']:.0f} prog={scores['task_progress']:.0f} "
                f"clean={scores['no_artifacts']:.0f} | {scores.get('reasoning','')[:60]}"
            )
            time.sleep(0.3)  # API 限流

        if not all_scores:
            return 0.0

        # 保存评分细节
        detail_path = os.path.join(output_dir, "vlm_scores.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_log, f, indent=2, ensure_ascii=False)

        # 取最低 bottom_k 个分数的平均值
        sorted_scores = sorted(all_scores)
        k = min(bottom_k, len(sorted_scores))
        bottom_avg = sum(sorted_scores[:k]) / k
        overall_avg = sum(all_scores) / len(all_scores)
        logger.info(
            f"VLM 评分统计: 全部{len(all_scores)}张均值={overall_avg:.1f} | "
            f"最低{k}张均值={bottom_avg:.1f} | 最低分={sorted_scores[0]:.0f} | "
            f"最高分={sorted_scores[-1]:.0f}"
        )
        return bottom_avg


# ============================================================
# 模块二: 训练执行器 (Training Runner)
# ============================================================
class TrainingRunner:
    """管理「配置生成 → 训练 → 推理」的完整流程。"""

    def __init__(
        self,
        base_dir: str,
        base_config_path: str,
        test_dir: str,
        gpus_per_node: int = 4,
        master_port: int = 25010,
        num_epochs: int = 5,
        inference_frame_interval: int = 6,
        inference_max_frames: int = 8,
        data_mode: str = "mixed",
        wandb_project: str = "VisualForesight",
    ):
        self.base_dir = Path(base_dir)
        self.base_config_path = Path(base_config_path)
        self.test_dir = test_dir
        self.gpus_per_node = gpus_per_node
        self.master_port = master_port
        self.default_num_epochs = num_epochs
        self.inference_frame_interval = inference_frame_interval
        self.inference_max_frames = inference_max_frames
        self.data_mode = data_mode
        self.wandb_project = wandb_project

        with open(self.base_config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

        # trial 配置文件目录（在 configs/ 下，与 possible_override_args 兼容）
        subdir = "automl_trials" if data_mode == "mixed" else "automl_pureorig_trials"
        self.configs_dir = self.base_dir / "configs" / subdir
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"TrainingRunner 初始化完成 | GPU={gpus_per_node} | num_epochs={num_epochs} | data_mode={data_mode}"
        )

    # ---- 配置生成 ----

    def generate_trial_config(self, trial_id: int, params: Dict[str, Any]) -> Path:
        """根据 Optuna 采样的超参数，生成 trial 专属 YAML 配置。"""
        config = copy.deepcopy(self.base_config)

        # 注入 Optuna 采样的超参数
        for key, value in params.items():
            config[key] = value

        # 修复 scheduler 兼容性：只有 cosine_with_min_lr 才需要 min_lr 参数，
        # linear / cosine / constant_with_warmup 不接受 min_lr
        sched = params.get("lr_scheduler_type", config.get("lr_scheduler_type", ""))
        if sched != "cosine_with_min_lr":
            config["lr_scheduler_kwargs"] = {}

        # 构造包含关键参数的 run_name，方便在 wandb 中辨识
        lr = params.get("learning_rate", config.get("learning_rate", 0))
        bs = params.get("per_device_train_batch_size",
                        config.get("per_device_train_batch_size", 0))
        ws = params.get("warmup_steps", config.get("warmup_steps", 0))
        sched_short = sched.replace("constant_with_warmup", "const") \
                           .replace("cosine_with_min_lr", "cos_min") \
                           .replace("cosine", "cos") \
                           .replace("linear", "lin")
        # ---- 根据 data_mode 调整数据源配置 ----
        if self.data_mode == "pure_orig":
            # 纯 orig 完整数据：移除 subtask 和 episode 过滤，保留 CoT
            config["subtask_data_path"] = ""
            config["filtered_episodes_path"] = ""
            config.setdefault("cot_json_path", "datasets/cot_by_episode.json")
        # mixed 模式：保留 base_config 中的 subtask_data_path 和 filtered_episodes_path

        mode_tag = "po" if self.data_mode == "pure_orig" else "mx"
        run_name = f"{mode_tag}_trial_{trial_id:03d}_lr{lr:.1e}_bs{bs}_{sched_short}_ws{ws}"

        ckpt_subdir = "automl_pureorig" if self.data_mode == "pure_orig" else "automl"
        config["output_dir"] = f"./checkpoints/{ckpt_subdir}/trial_{trial_id:03d}"
        config["run_name"] = run_name
        config["num_train_epochs"] = self.default_num_epochs
        config["max_steps"] = -1                      # 不限制步数，由 epoch 控制
        config["save_strategy"] = "steps"             # 每 N 步保存
        config["save_steps"] = 50                     # 每 50 步保存一次
        config["save_total_limit"] = 2
        config["report_to"] = "wandb"                 # 开启 wandb 追踪
        config["overwrite_output_dir"] = True

        config_path = self.configs_dir / f"trial_{trial_id:03d}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Trial {trial_id} 配置已生成: {config_path}")
        return config_path


    # ---- GPU 进程清理 ----

    @staticmethod
    def _kill_process_tree(proc):
        """通过进程组杀死进程及其所有子进程。"""
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(3)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        except (ProcessLookupError, PermissionError, OSError):
            pass

    @staticmethod
    def _cleanup_gpu_processes():
        """清理残留在 GPU 上的 train.py / torchrun 僵尸进程。"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
                    if "train.py" in cmdline or "batch_inference" in cmdline:
                        os.kill(pid, signal.SIGKILL)
                        logger.warning(f"已清理残留 GPU 进程: PID={pid}")
                except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                    pass
        except Exception as e:
            logger.warning(f"GPU 进程清理时出错（不影响后续运行）: {e}")


    # ---- 训练 ----

    def run_training(self, trial_id: int, config_path: Path) -> Optional[str]:
        """调用 torchrun 执行训练，返回 checkpoint 路径；失败返回 None。"""
        # possible_override_args 会自动加 "configs/" 前缀
        config_subdir = "automl_pureorig_trials" if self.data_mode == "pure_orig" else "automl_trials"
        relative_config = f"{config_subdir}/{config_path.name}"

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.gpus_per_node}",
            "--master_port", str(self.master_port),
            "train.py",
            "--config_file", relative_config,
        ]

        logger.info(f"[训练] Trial {trial_id}: {' '.join(cmd)}")

        with open(config_path, "r") as f:
            trial_config = yaml.safe_load(f)
        output_dir = trial_config["output_dir"]
        run_name = trial_config.get("run_name", f"trial_{trial_id:03d}")

        proc = None
        try:
            train_env = os.environ.copy()
            train_env["WANDB_PROJECT"] = self.wandb_project
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=train_env,
                start_new_session=True,  # 新进程组，方便整体清理
            )
            stdout, stderr = proc.communicate()  # 不限时，等待训练自然结束
            if proc.returncode != 0:
                stderr = stderr or ""
                if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
                    logger.error(f"Trial {trial_id} OOM！此参数组合无法在当前 GPU 运行")
                else:
                    logger.error(
                        f"Trial {trial_id} 训练失败 (code={proc.returncode})\n"
                        f"stderr 尾部:\n{stderr[-800:]}"
                    )
                self._kill_process_tree(proc)
                self._cleanup_gpu_processes()
                return None
        except Exception as e:
            logger.error(f"Trial {trial_id} 训练异常: {e}")
            self._kill_process_tree(proc)
            self._cleanup_gpu_processes()
            return None

        # 查找 checkpoint
        ckpt_parent = Path(output_dir) / run_name
        if not ckpt_parent.is_dir():
            logger.error(f"训练输出目录不存在: {ckpt_parent}")
            return None

        ckpts = sorted(glob.glob(str(ckpt_parent / "checkpoint-*")))
        if ckpts:
            logger.info(f"找到 checkpoint: {ckpts[-1]}")
            return ckpts[-1]

        # 也许模型直接保存在 run_name 目录
        if any(f.endswith((".safetensors", ".pt", ".bin"))
               for f in os.listdir(ckpt_parent)):
            return str(ckpt_parent)

        logger.error(f"未找到 checkpoint: {ckpt_parent}")
        return None

    # ---- 推理 ----

    def run_inference(self, checkpoint_path: str, trial_id: int) -> Optional[str]:
        """调用 batch_inference.py 生成预测图像，返回输出目录；失败返回 None。"""
        results_subdir = "batch_results_automl_pureorig" if self.data_mode == "pure_orig" else "batch_results_automl"
        output_dir = str(
            PROJECT_ROOT / results_subdir / f"trial_{trial_id:03d}"
        )
        cmd = [
            "python", "batch_inference.py",
            "--checkpoint_path", checkpoint_path,
            "--test_dir", self.test_dir,
            "--output_dir", output_dir,
            "--frame_interval", str(self.inference_frame_interval),
            "--max_frames_per_video", str(self.inference_max_frames),
        ]
        logger.info(f"[推理] Trial {trial_id}: {' '.join(cmd[-6:])}")

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(
                    f"推理失败 (code={proc.returncode})\n"
                    f"stderr 尾部:\n{(stderr or '')[-500:]}"
                )
                self._kill_process_tree(proc)
                self._cleanup_gpu_processes()
                return None
        except Exception as e:
            logger.error(f"推理异常: {e}")
            self._kill_process_tree(proc)
            self._cleanup_gpu_processes()
            return None

        pred_files = glob.glob(
            os.path.join(output_dir, "**", "*_predicted.png"), recursive=True
        )
        if not pred_files:
            logger.error("推理未生成任何预测图像")
            return None

        logger.info(f"推理完成，共 {len(pred_files)} 张预测图")
        return output_dir


# ============================================================
# 模块三: Optuna 优化主循环（流水线式 ask/tell）
# ============================================================
def sample_params(trial) -> dict:
    """从 Optuna trial 中采样超参数。"""
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1.4e-5, 1.5e-5, log=True
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8,16]
        ),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 150, step=50),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            ["constant_with_warmup", "cosine", "cosine_with_min_lr", "linear"],
        ),
    }


def train_and_infer(runner, tid: int, params: dict):
    """执行训练 + 推理，返回 (checkpoint_path, inference_output_dir)。"""
    try:
        config_path = runner.generate_trial_config(tid, params)
        checkpoint = runner.run_training(tid, config_path)
    except Exception as e:
        logger.error(f"训练阶段异常:\n{traceback.format_exc()}")
        return None, None

    if checkpoint is None:
        return None, None

    try:
        inference_output = runner.run_inference(checkpoint, tid)
    except Exception as e:
        logger.error(f"推理阶段异常:\n{traceback.format_exc()}")
        return checkpoint, None

    return checkpoint, inference_output


def run_study_pipeline(study, runner, evaluator, n_trials: int):
    """流水线式 Optuna ask/tell 主循环：VLM 评分与下一个 trial 的训练并行。

    时间线示意：
      Trial N:  ====[训练+推理]====|--[VLM评分(后台)]--|
      Trial N+1:                   ====[训练+推理]====|--[VLM评分(后台)]--|
                                   ↑ 这里 GPU 空闲即开始，不等评分

    因为训练占 GPU、评分只占 API 带宽，两者完全并行。
    """
    vlm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm_scorer")
    pending_score = None   # (trial_number, trial_obj, Future)

    def _finish_pending():
        """等待挂起的 VLM 评分完成并 tell Optuna。"""
        nonlocal pending_score
        if pending_score is None:
            return
        prev_tid, prev_trial, future = pending_score
        try:
            score = future.result(timeout=3600)
        except Exception as e:
            logger.error(f"Trial {prev_tid} VLM 评分超时/异常: {e}")
            score = 0.0
        study.tell(prev_trial, score)
        logger.info(f">>> Trial {prev_tid} 评分已回传 Optuna: {score:.1f}/100")
        pending_score = None

    for i in range(n_trials):
        # 如果上一轮评分已完成（通常训练时间 >> 评分时间），先回收
        if pending_score is not None and pending_score[2].done():
            _finish_pending()

        # ---- ask ----
        trial = study.ask()
        tid = trial.number
        sep = '=' * 60; logger.info(f"\n{sep}\n>>> Trial {tid} 开始 ({i+1}/{n_trials})\n{sep}")

        params = sample_params(trial)
        logger.info(f"Trial {tid} 参数: {json.dumps(params, indent=2)}")

        # ---- 训练 + 推理 ----
        checkpoint, inference_output = train_and_infer(runner, tid, params)

        if inference_output is None:
            logger.warning(f"Trial {tid} 训练/推理失败 → 返回 0 分")
            study.tell(trial, 0.0)
            continue

        # ---- 确保上一个评分已完成再提交新的 ----
        _finish_pending()

        # ---- 异步提交 VLM 评分 ----
        future = vlm_pool.submit(evaluator.score_trial_output, inference_output)
        pending_score = (tid, trial, future)
        logger.info(f"Trial {tid} VLM 评分已提交后台，下一个 trial 的训练可立即开始")

    # ---- 清理：等待最后一个评分完成 ----
    _finish_pending()
    vlm_pool.shutdown(wait=True)


# ============================================================
# 干跑测试
# ============================================================
def run_dry_test(runner: TrainingRunner, evaluator: VLMEvaluator, args):
    """验证各模块的基本功能（不实际训练）。"""
    logger.info("\n" + "=" * 60)
    logger.info(">>> 干跑测试模式（Dry Run）")
    logger.info("=" * 60)

    # --- 1. 测试配置生成 ---
    logger.info("\n[1/3] 测试配置生成...")
    test_params = {
        "learning_rate": 1.4e-5,
        "per_device_train_batch_size": 8,
        "warmup_steps": 100,
        "lr_scheduler_type": "cosine",
    }
    config_path = runner.generate_trial_config(999, test_params)
    with open(config_path, "r") as f:
        content = f.read()
    logger.info(f"生成的配置:\n{content}")
    logger.info("✓ 配置生成测试通过\n")

    # --- 2. 测试 VLM 评分（用现有的 batch_results 图做测试）---
    logger.info("[2/3] 测试 VLM 评分...")
    existing_results = None
    for d in ["batch_results_3", "batch_results_2"]:
        p = PROJECT_ROOT / d
        if p.is_dir() and glob.glob(str(p / "**/*_predicted.png"), recursive=True):
            existing_results = str(p)
            break

    if existing_results:
        logger.info(f"使用已有推理结果测试 VLM: {existing_results}")
        try:
            score = evaluator.score_trial_output(existing_results, bottom_k=2, max_pairs=3)
            logger.info(f"✓ VLM 评分测试通过，得分: {score:.1f}/100")
        except Exception as e:
            logger.error(f"✗ VLM 评分测试失败: {e}\n{traceback.format_exc()}")
    else:
        logger.warning("未找到已有推理结果，跳过 VLM 评分测试")

    # --- 3. 检查训练命令 ---
    logger.info("\n[3/3] 训练命令预览...")
    logger.info(
        f"  torchrun --nproc_per_node={runner.gpus_per_node} "
        f"--master_port={runner.master_port} train.py "
        f"--config_file {'automl_pureorig_trials' if runner.data_mode == 'pure_orig' else 'automl_trials'}/trial_999.yaml"
    )
    logger.info("✓ 训练命令构造正确\n")

    # 清理临时配置
    os.remove(config_path)

    logger.info("=" * 60)
    logger.info(">>> 干跑测试全部完成！")
    logger.info("=" * 60)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Agentic AutoML: Optuna + 豆包 VLM 自动调参系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 干跑测试\n"
            "  python automl_optuna.py --vlm_model ep-XXXX --dry_run\n\n"
            "  # 正式运行 20 轮搜索\n"
            "  python automl_optuna.py --vlm_model ep-XXXX --n_trials 20\n"
        ),
    )
    parser.add_argument(
        "--vlm_model", type=str, required=True,
        help="豆包视觉模型 endpoint ID（如 ep-20241201xxxxx），"
             "需在火山引擎控制台创建推理接入点",
    )
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna 搜索轮数")
    parser.add_argument("--num_epochs", type=int, default=5, help="每轮训练 epoch 数")
    parser.add_argument("--gpus", type=int, default=8, help="GPU 数量")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG), help="基础配置文件路径"
    )
    parser.add_argument(
        "--test_dir", type=str, default=str(DEFAULT_TEST_DIR), help="测试视频目录"
    )
    parser.add_argument("--master_port", type=int, default=25010, help="torchrun port")
    parser.add_argument("--dry_run", action="store_true", help="干跑测试（不实际训练）")
    parser.add_argument(
        "--data_mode", type=str, default="mixed", choices=["mixed", "pure_orig"],
        help="数据模式: mixed=精选orig+subtask(默认), pure_orig=纯完整orig数据+CoT",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="wandb 项目名（默认: mixed用VisualForesight, pure_orig用VisualForesight_PureOrig）",
    )
    parser.add_argument(
        "--study_name", type=str, default=None,
        help="Optuna study 名称（默认: mixed用foreact_automl, pure_orig用foreact_automl_pureorig）",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage（默认: mixed用sqlite:///automl.db, pure_orig用sqlite:///automl_pureorig.db）",
    )
    args = parser.parse_args()

    # ---- 根据 data_mode 设置默认值 ----
    if args.data_mode == "pure_orig":
        if args.wandb_project is None:
            args.wandb_project = "VisualForesight_PureOrig"
        if args.study_name is None:
            args.study_name = "foreact_automl_pureorig"
        if args.storage is None:
            args.storage = "sqlite:///automl_pureorig.db"
    else:
        if args.wandb_project is None:
            args.wandb_project = "VisualForesight"
        if args.study_name is None:
            args.study_name = "foreact_automl"
        if args.storage is None:
            args.storage = "sqlite:///automl.db"

    # ---- 环境检查 ----
    api_key = os.environ.get("VOLCENKEY")
    if not api_key:
        logger.error("请先设置环境变量: export VOLCENKEY='your-api-key'")
        sys.exit(1)

    if not os.path.isdir(args.test_dir):
        logger.error(f"测试目录不存在: {args.test_dir}")
        sys.exit(1)

    test_videos = glob.glob(os.path.join(args.test_dir, "*.mp4"))
    if not test_videos:
        logger.error(f"测试目录中无 .mp4 文件: {args.test_dir}")
        sys.exit(1)
    logger.info(f"找到 {len(test_videos)} 个测试视频")

    # ---- 依赖检查 ----
    try:
        import optuna  # noqa: F811
    except ImportError:
        logger.error("缺少 optuna，请运行: pip install optuna")
        sys.exit(1)

    # ---- 初始化组件 ----
    evaluator = VLMEvaluator(api_key=api_key, model_id=args.vlm_model)
    runner = TrainingRunner(
        base_dir=str(PROJECT_ROOT),
        base_config_path=args.config,
        test_dir=args.test_dir,
        gpus_per_node=args.gpus,
        master_port=args.master_port,
        num_epochs=args.num_epochs,
        data_mode=args.data_mode,
        wandb_project=args.wandb_project,
    )

    # ---- 干跑模式 ----
    if args.dry_run:
        run_dry_test(runner, evaluator, args)
        return

    # ---- 正式 Optuna Study ----
    storage = args.storage if args.storage and args.storage.lower() != "none" else None
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"启动 Optuna Study: {args.study_name}")
    logger.info(f"  总轮数:        {args.n_trials}")
    logger.info(f"  每轮训练轮数:  {args.num_epochs} epochs")
    logger.info(f"  GPU 数量:      {args.gpus}")
    logger.info(f"  存储:          {storage or '内存'}")
    logger.info(f"  数据模式:      {args.data_mode}")
    logger.info(f"  wandb项目:     {args.wandb_project}")
    logger.info(f"  评分模式:      全部图像评分 → 最低10张均值")
    logger.info(f"  并行模式:      VLM评分与下一trial训练并行")
    logger.info(f"{'='*60}\n")

    run_study_pipeline(study, runner, evaluator, args.n_trials)

    # ---- 输出最佳结果 ----
    logger.info("\n" + "=" * 60)
    logger.info(">>> Optuna Study 完成！")
    if study.best_trial:
        logger.info(f"  最佳 Trial:  #{study.best_trial.number}")
        logger.info(f"  最佳得分:    {study.best_value:.1f}/100")
        logger.info(f"  最佳参数:    {json.dumps(study.best_params, indent=4)}")
    else:
        logger.warning("  没有成功完成的 trial")
    logger.info("=" * 60)

    # 保存完整结果到 JSON
    results_name = "automl_pureorig_results.json" if args.data_mode == "pure_orig" else "automl_results.json"
    results_path = PROJECT_ROOT / results_name
    results = {
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_score": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"完整结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
