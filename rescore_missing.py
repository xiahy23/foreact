"""
对 batch_results_automl 中缺少 vlm_scores.json 的 trial 目录重新评分。

用法:
    python rescore_missing.py --vlm_model ep-XXXXXXXX
    python rescore_missing.py --vlm_model ep-XXXXXXXX --results_dir /path/to/batch_results_automl
    python rescore_missing.py --vlm_model ep-XXXXXXXX --dry_run

依赖环境变量:
    VOLCENKEY  — 火山引擎 API Key
"""

import argparse
import base64
import glob
import io
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


class VLMEvaluator:
    SCORING_PROMPT = (
        '你是一个专业的机器人视觉预测模型评审专家。\n'
        '我会给你两张图片：\n'
        '- 第一张是"输入图"（机械臂操作场景的当前状态）\n'
        '- 第二张是"预测图"（模型预测的下一步状态）\n'
        '- 当前任务是："{task_prompt}"\n\n'
        '请你严格按照以下 5 个维度对预测图进行打分（每个维度 0-20 分）：\n\n'
        '1. **scene_consistency**（场景一致性，0-20）\n'
        '2. **gripper_clarity**（夹爪清晰度，0-20）\n'
        '3. **physics_plausibility**（物理合理性，0-20）\n'
        '4. **task_progress**（任务进展合理性，0-20）\n'
        '5. **no_artifacts**（无杂物/无偷懒，0-20）\n\n'
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

    @staticmethod
    def _pil_to_base64_uri(img: "Image.Image", fmt: str = "JPEG", quality: int = 85) -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    def _parse_scores(self, raw_text: str) -> Optional[Dict[str, Any]]:
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
        required = ["scene_consistency", "gripper_clarity",
                    "physics_plausibility", "task_progress", "no_artifacts"]
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

    def score_image_pair(self, input_img, pred_img, task_prompt: str, retry: int = 3) -> Dict[str, Any]:
        input_uri = self._pil_to_base64_uri(input_img)
        pred_uri = self._pil_to_base64_uri(pred_img)
        prompt = self.SCORING_PROMPT.replace("{task_prompt}", task_prompt)
        for attempt in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": input_uri}},
                        {"type": "image_url", "image_url": {"url": pred_uri}},
                    ]}],
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

    def score_trial_output(self, output_dir: str, bottom_k: int = 10) -> float:
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
            logger.error(f"  在 {output_dir} 中未找到图像对，跳过")
            return 0.0

        logger.info(f"  共找到 {len(pairs)} 组图像对，开始评分...")
        all_scores: List[float] = []
        detail_log: List[Dict] = []

        for i, (inp_path, pred_path, task_prompt) in enumerate(pairs):
            try:
                input_img = Image.open(inp_path).convert("RGB")
                pred_img = Image.open(pred_path).convert("RGB")
            except Exception as e:
                logger.warning(f"  读取图像失败: {e}")
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
            time.sleep(0.3)

        if not all_scores:
            return 0.0

        detail_path = os.path.join(output_dir, "vlm_scores.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_log, f, indent=2, ensure_ascii=False)
        logger.info(f"  评分已保存到: {detail_path}")

        sorted_scores = sorted(all_scores)
        k = min(bottom_k, len(sorted_scores))
        bottom_avg = sum(sorted_scores[:k]) / k
        overall_avg = sum(all_scores) / len(all_scores)
        logger.info(
            f"  统计: 全部{len(all_scores)}张均值={overall_avg:.1f} | "
            f"最低{k}张均值={bottom_avg:.1f} | 最低={sorted_scores[0]:.0f} 最高={sorted_scores[-1]:.0f}"
        )
        return bottom_avg


def find_missing_trials(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        logger.error(f"目录不存在: {results_dir}")
        return []
    missing = []
    for name in sorted(os.listdir(results_dir)):
        trial_dir = os.path.join(results_dir, name)
        if not os.path.isdir(trial_dir):
            continue
        if not os.path.isfile(os.path.join(trial_dir, "vlm_scores.json")):
            missing.append(trial_dir)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="对 batch_results_automl 中缺少评分的 trial 目录重新用豆包 VLM 评分"
    )
    parser.add_argument("--vlm_model", type=str, required=True,
                        help="豆包视觉模型 endpoint ID")
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_results_automl"),
                        help="batch_results_automl 目录路径")
    parser.add_argument("--bottom_k", type=int, default=10,
                        help="取最低 k 个分数求均值（默认 10）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只列出缺少评分的 trial，不调用 API")
    args = parser.parse_args()

    missing = find_missing_trials(args.results_dir)
    if not missing:
        logger.info("所有 trial 均已有 vlm_scores.json，无需补评。")
        return

    logger.info(f"发现 {len(missing)} 个缺少评分的 trial：")
    for d in missing:
        logger.info(f"  {os.path.basename(d)}")

    if args.dry_run:
        logger.info("[dry_run] 不调用 API，退出。")
        return

    api_key = os.environ.get("VOLCENKEY")
    if not api_key:
        logger.error("环境变量 VOLCENKEY 未设置，请先执行: export VOLCENKEY=<your_api_key>")
        sys.exit(1)

    evaluator = VLMEvaluator(api_key=api_key, model_id=args.vlm_model)

    results_summary: Dict[str, float] = {}
    for i, trial_dir in enumerate(missing):
        trial_name = os.path.basename(trial_dir)
        logger.info(f"\n[{i+1}/{len(missing)}] 开始评分: {trial_name}")
        score = evaluator.score_trial_output(trial_dir, bottom_k=args.bottom_k)
        results_summary[trial_name] = score
        logger.info(f"  => bottom_{args.bottom_k}_avg = {score:.2f}")

    logger.info("\n==================== 补评汇总 ====================")
    for trial_name, score in sorted(results_summary.items()):
        logger.info(f"  {trial_name}: {score:.2f}")
    logger.info("===================================================")
    logger.info("全部补评完成。")


if __name__ == "__main__":
    main()
