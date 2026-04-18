#!/usr/bin/env python3
"""
F-exp02 AutoML: all_eval + bridge, explore per-device batch size (no accum).

Search space:
  - per_device_train_batch_size: [16, 32, 64, 128, 256, 512]
  - gradient_accumulation_steps: FIXED at 1 (no accumulation)
  - learning_rate:               log-uniform [3e-6, 3e-5]
  - lr_scheduler_type:           [cosine_with_min_lr, cosine, linear, constant_with_warmup]
  - warmup_steps:                [50, 100, ..., 500]
  - unfreeze_mllm:               [False, True]
  - (if cosine_with_min_lr) min_lr_ratio: log-uniform [0.01, 0.2]

Each trial config written to configs/F-exp02-automl-alleval-trials/trial_XXX.yaml
with a header describing its hyperparameters.

Checkpointing: save_strategy=epoch, save_total_limit=100 -> keep all epoch ckpts.

Usage:
  export VOLCENKEY="..."
  python automl_alleval.py --vlm_model doubao-seed-2-0-lite-260215 --n_trials 12
  python automl_alleval.py --vlm_model doubao-seed-2-0-lite-260215 --dry_run
"""
import argparse
import copy
import glob
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import yaml

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("F-exp02-automl-alleval.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("F-exp02")

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "F-exp02-automl-alleval-base.yaml"
TEST_DIR = PROJECT_ROOT / "test"
EXP_TAG = "F-exp02"

# Reuse VLM evaluator
sys.path.insert(0, str(PROJECT_ROOT))
from automl_optuna import VLMEvaluator, VIDEO_PROMPTS  # noqa: E402


# ============================================================
# Parameter search space
# ============================================================
def sample_params(trial) -> dict:
    lr = trial.suggest_float("learning_rate", 3e-6, 3e-5, log=True)
    scheduler = trial.suggest_categorical(
        "lr_scheduler_type",
        ["cosine_with_min_lr", "cosine", "linear", "constant_with_warmup"],
    )
    warmup = trial.suggest_int("warmup_steps", 50, 500, step=50)
    # per_device batch size — NO accumulation; trying the full requested range.
    batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32, 64, 128, 256, 512]
    )
    unfreeze_mllm = trial.suggest_categorical("unfreeze_mllm", [False, True])

    params = {
        "learning_rate": lr,
        "lr_scheduler_type": scheduler,
        "warmup_steps": warmup,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "unfreeze_mllm": unfreeze_mllm,
    }
    if scheduler == "cosine_with_min_lr":
        min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.01, 0.2, log=True)
        params["min_lr"] = lr * min_lr_ratio
        params["min_lr_ratio"] = min_lr_ratio
    return params


# ============================================================
# Training runner
# ============================================================
class AllEvalTrainingRunner:
    def __init__(
        self,
        base_config_path: str,
        test_dir: str,
        num_gpus: int = 8,
        num_epochs: int = 10,
        wandb_project: str = "VisualForesight_F_exp02",
    ):
        self.base_config_path = Path(base_config_path)
        self.test_dir = test_dir
        self.num_gpus = num_gpus
        self.num_epochs = num_epochs
        self.wandb_project = wandb_project

        with open(self.base_config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

        self.configs_dir = PROJECT_ROOT / "configs" / f"{EXP_TAG}-automl-alleval-trials"
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def generate_trial_config(self, trial_id: int, params: dict) -> Path:
        config = copy.deepcopy(self.base_config)

        for key in [
            "learning_rate", "lr_scheduler_type", "warmup_steps",
            "per_device_train_batch_size", "gradient_accumulation_steps",
        ]:
            if key in params:
                config[key] = params[key]

        sched = params.get("lr_scheduler_type", "cosine_with_min_lr")
        if sched == "cosine_with_min_lr" and "min_lr" in params:
            config["lr_scheduler_kwargs"] = {"min_lr": params["min_lr"]}
        else:
            config["lr_scheduler_kwargs"] = {}

        if params.get("unfreeze_mllm", False):
            config["modules_to_freeze"] = ["vae"]
        else:
            config["modules_to_freeze"] = ["vae", "mllm_backbone"]

        lr = params["learning_rate"]
        bs = params["per_device_train_batch_size"]
        sched_short = (
            sched.replace("constant_with_warmup", "const")
                 .replace("cosine_with_min_lr", "cos_min")
                 .replace("cosine", "cos")
                 .replace("linear", "lin")
        )
        unfreeze_tag = "uf" if params.get("unfreeze_mllm") else "fr"

        run_name = (
            f"{EXP_TAG}_t{trial_id:03d}"
            f"_bs{bs}_lr{lr:.1e}_{sched_short}_w{params['warmup_steps']}_{unfreeze_tag}"
        )
        config["run_name"] = run_name
        config["output_dir"] = f"./checkpoints/{EXP_TAG}-automl-alleval/trial_{trial_id:03d}"
        config["num_train_epochs"] = float(self.num_epochs)
        config["max_steps"] = -1
        # 1 checkpoint per epoch, retain all (10 epoch ckpts)
        config["save_strategy"] = "epoch"
        config["save_total_limit"] = 100
        config["report_to"] = "wandb"
        config["overwrite_output_dir"] = True

        # -------- Documentation header (YAML comment) --------
        header_lines = [
            f"# =============================================================",
            f"# {EXP_TAG} trial {trial_id:03d}  (run_name: {run_name})",
            f"# -------------------------------------------------------------",
            f"# Generated by automl_alleval.py — do NOT edit by hand.",
            f"# ",
            f"# Hyperparameters (sampled by Optuna):",
            f"#   per_device_train_batch_size : {bs}",
            f"#   gradient_accumulation_steps : 1  (FIXED — no accumulation)",
            f"#   effective batch (global)    : {bs} * {self.num_gpus} gpus = {bs * self.num_gpus}",
            f"#   learning_rate               : {lr:.3e}",
            f"#   lr_scheduler_type           : {sched}",
            f"#   warmup_steps                : {params['warmup_steps']}",
            f"#   unfreeze_mllm               : {params.get('unfreeze_mllm', False)}",
        ]
        if sched == "cosine_with_min_lr":
            header_lines.append(f"#   min_lr_ratio                : {params.get('min_lr_ratio', 0):.4f}")
            header_lines.append(f"#   min_lr                      : {params.get('min_lr', 0):.3e}")
        header_lines += [
            f"# ",
            f"# Saving: save_strategy=epoch, save_total_limit=100",
            f"#         -> 1 ckpt per epoch, all {int(self.num_epochs)} retained.",
            f"# Resume: ./foreact-pretrained (same init as F-exp01)",
            f"# =============================================================",
            "",
        ]
        header = "\n".join(header_lines)

        config_path = self.configs_dir / f"trial_{trial_id:03d}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

        logger.info(f"Trial {trial_id} config: {config_path}")
        return config_path

    @staticmethod
    def _kill_process_tree(proc):
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
                        logger.warning(f"Cleaned GPU process: PID={pid}")
                except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                    pass
        except Exception as e:
            logger.warning(f"GPU cleanup error (non-fatal): {e}")

    def run_training(self, trial_id: int, config_path: Path) -> Optional[str]:
        relative_config = f"{EXP_TAG}-automl-alleval-trials/{config_path.name}"
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(self.num_gpus),
            "--mixed_precision", "bf16",
            "train.py",
            "--config_file", relative_config,
        ]
        logger.info(f"[Training] Trial {trial_id}: {' '.join(cmd)}")

        with open(config_path, "r") as f:
            trial_config = yaml.safe_load(f)
        output_dir = trial_config["output_dir"]
        run_name = trial_config.get("run_name", f"trial_{trial_id:03d}")

        proc = None
        try:
            env = os.environ.copy()
            env["WANDB_PROJECT"] = self.wandb_project
            env["FORCE_VIDEO_BACKEND"] = "pyav"
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
            env["HF_HUB_ETAG_TIMEOUT"] = "30"

            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, start_new_session=True,
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                stderr = stderr or ""
                if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
                    logger.error(f"Trial {trial_id} OOM! bs={trial_config['per_device_train_batch_size']}")
                else:
                    logger.error(
                        f"Trial {trial_id} training failed (code={proc.returncode})\n"
                        f"stderr tail:\n{stderr[-1200:]}"
                    )
                self._kill_process_tree(proc)
                self._cleanup_gpu_processes()
                return None
        except Exception as e:
            logger.error(f"Trial {trial_id} training exception: {e}")
            self._kill_process_tree(proc)
            self._cleanup_gpu_processes()
            return None

        ckpt_parent = Path(output_dir) / run_name
        if not ckpt_parent.is_dir():
            ckpt_parent = Path(output_dir)

        if ckpt_parent.is_dir():
            ckpts = sorted(glob.glob(str(ckpt_parent / "checkpoint-*")))
            if ckpts:
                logger.info(f"Trial {trial_id} -> found {len(ckpts)} ckpts, using last: {ckpts[-1]}")
                return ckpts[-1]
            if any(f.endswith((".safetensors", ".pt", ".bin")) for f in os.listdir(ckpt_parent)):
                return str(ckpt_parent)

        logger.error(f"No checkpoint found in: {output_dir}")
        return None

    def run_inference(self, checkpoint_path: str, trial_id: int) -> Optional[str]:
        output_dir = str(PROJECT_ROOT / f"batch_results_{EXP_TAG}_automl_alleval" / f"trial_{trial_id:03d}")
        cmd = [
            "python", "batch_inference.py",
            "--checkpoint_path", checkpoint_path,
            "--test_dir", self.test_dir,
            "--output_dir", output_dir,
            "--frame_interval", "6",
            "--max_frames_per_video", "8",
        ]
        logger.info(f"[Inference] Trial {trial_id}: {' '.join(cmd[-6:])}")

        proc = None
        try:
            env = os.environ.copy()
            env["FORCE_VIDEO_BACKEND"] = "pyav"
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, start_new_session=True,
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Inference failed (code={proc.returncode})\nstderr:\n{(stderr or '')[-800:]}")
                self._kill_process_tree(proc)
                self._cleanup_gpu_processes()
                return None
        except Exception as e:
            logger.error(f"Inference exception: {e}")
            self._kill_process_tree(proc)
            self._cleanup_gpu_processes()
            return None

        pred_files = glob.glob(os.path.join(output_dir, "**", "*_predicted.png"), recursive=True)
        if not pred_files:
            logger.error("Inference produced no prediction images")
            return None
        logger.info(f"Inference done: {len(pred_files)} prediction images")
        return output_dir


# ============================================================
# Optuna study
# ============================================================
def run_study(args):
    import optuna
    from concurrent.futures import ThreadPoolExecutor

    api_key = os.environ.get("VOLCENKEY")
    if not api_key and not args.dry_run:
        logger.error("Set VOLCENKEY env var first!")
        sys.exit(1)

    evaluator = VLMEvaluator(api_key=api_key, model_id=args.vlm_model) if api_key else None
    runner = AllEvalTrainingRunner(
        base_config_path=str(BASE_CONFIG),
        test_dir=args.test_dir,
        num_gpus=args.gpus,
        num_epochs=args.num_epochs,
        wandb_project=args.wandb_project,
    )

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        test_params = sample_params(optuna.trial.FixedTrial({
            "learning_rate": 1e-5,
            "lr_scheduler_type": "cosine_with_min_lr",
            "warmup_steps": 100,
            "per_device_train_batch_size": 32,
            "unfreeze_mllm": False,
            "min_lr_ratio": 0.1,
        }))
        config_path = runner.generate_trial_config(999, test_params)
        with open(config_path) as f:
            logger.info(f"Generated config:\n{f.read()}")
        logger.info("Dry run done (generated config left at %s)", config_path)
        return

    storage = f"sqlite:///{EXP_TAG}-automl-alleval.db"
    study = optuna.create_study(
        study_name=f"foreact_{EXP_TAG}_automl_alleval",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Count already-finished trials so restart only runs remaining ones
    import optuna as _optuna
    done_trials = [
        t for t in study.trials
        if t.state in (_optuna.trial.TrialState.COMPLETE, _optuna.trial.TrialState.PRUNED)
    ]
    n_done = len(done_trials)
    n_remaining = max(0, args.n_trials - n_done)

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {EXP_TAG} Optuna Study")
    logger.info(f"  Total trials requested : {args.n_trials}")
    logger.info(f"  Already completed      : {n_done}")
    logger.info(f"  Remaining to run       : {n_remaining}")
    logger.info(f"  Epochs:  {args.num_epochs}")
    logger.info(f"  GPUs:    {args.gpus}")
    logger.info(f"  Storage: {storage}")
    logger.info(f"{'='*60}\n")

    if n_remaining == 0:
        logger.info("All trials already completed. Nothing to do.")
        return

    vlm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm_scorer")
    pending_score = None

    def _finish_pending():
        nonlocal pending_score
        if pending_score is None:
            return
        prev_tid, prev_trial, future = pending_score
        try:
            score = future.result(timeout=3600)
        except Exception as e:
            logger.error(f"Trial {prev_tid} VLM scoring failed: {e}")
            score = 0.0
        study.tell(prev_trial, score)
        logger.info(f">>> Trial {prev_tid} score reported: {score:.1f}/100")
        pending_score = None

    for i in range(n_remaining):
        if pending_score is not None and pending_score[2].done():
            _finish_pending()

        trial = study.ask()
        tid = trial.number
        logger.info(f"\n{'='*60}\n>>> Trial {tid} ({i+1}/{n_remaining}, total done after: {n_done+i+1}/{args.n_trials})\n{'='*60}")

        params = sample_params(trial)
        logger.info(f"Trial {tid} params: {json.dumps({k: str(v) for k, v in params.items()}, indent=2)}")

        try:
            config_path = runner.generate_trial_config(tid, params)
            checkpoint = runner.run_training(tid, config_path)
        except Exception:
            logger.error(f"Training failed:\n{traceback.format_exc()}")
            checkpoint = None

        if checkpoint is None:
            logger.warning(f"Trial {tid} training failed -> score 0")
            study.tell(trial, 0.0)
            continue

        try:
            inference_output = runner.run_inference(checkpoint, tid)
        except Exception:
            logger.error(f"Inference failed:\n{traceback.format_exc()}")
            inference_output = None

        if inference_output is None:
            logger.warning(f"Trial {tid} inference failed -> score 0")
            study.tell(trial, 0.0)
            continue

        _finish_pending()
        future = vlm_pool.submit(evaluator.score_trial_output, inference_output)
        pending_score = (tid, trial, future)
        logger.info(f"Trial {tid} VLM scoring submitted (async)")

    _finish_pending()
    vlm_pool.shutdown(wait=True)

    logger.info(f"\n{'='*60}\n>>> Study Complete!")
    if study.best_trial:
        logger.info(f"  Best Trial:  #{study.best_trial.number}")
        logger.info(f"  Best Score:  {study.best_value:.1f}/100")
        logger.info(f"  Best Params: {json.dumps(study.best_params, indent=4)}")
    else:
        logger.warning("  No successful trials")
    logger.info("=" * 60)

    results = {
        "exp_tag": EXP_TAG,
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_score": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    results_path = PROJECT_ROOT / f"{EXP_TAG}-automl-alleval-results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description=f"{EXP_TAG} AutoML for ForeAct all_eval+bridge")
    parser.add_argument("--vlm_model", type=str, required=True, help="VLM model ID for scoring")
    parser.add_argument("--n_trials", type=int, default=12, help="Number of Optuna trials")
    parser.add_argument("--num_epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--test_dir", type=str, default=str(TEST_DIR), help="Test video dir")
    parser.add_argument("--dry_run", action="store_true", help="Generate a sample trial config and exit")
    parser.add_argument("--wandb_project", type=str, default="VisualForesight_F_exp02")
    args = parser.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
