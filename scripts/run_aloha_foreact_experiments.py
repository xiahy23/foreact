#!/usr/bin/env python3
"""Long-running ForeAct experiment runner for Aloha banana/cube main-view videos.

The runner creates F-expNN configs, commits each experiment config before launch,
then executes experiments sequentially. If there is no ForeAct experiment left to
run, it falls back to the requested starVLA keep-busy job.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
FINETUNE_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "finetune"
LOG_DIR = PROJECT_ROOT / "logs" / "aloha_fexp"
STATE_PATH = PROJECT_ROOT / "logs" / "aloha_fexp_state.json"
SPLIT_ROOT = PROJECT_ROOT / "datasets" / "aloha_fexp_roots"
BANANA_DATASET = PROJECT_ROOT / "datasets" / "aloha_banana"
CUBE_DATASET = PROJECT_ROOT / "datasets" / "aloha_cube"
STARVLA_SCRIPT = Path("/media/raid/workspace/xiahongyu/Agent-VLA/playground/Datasets/act_cot.py")

LOCAL_MLLM = "/media/raid/workspace/xiahongyu/.cache/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
LOCAL_SANA = "/media/raid/workspace/xiahongyu/.cache/hub/models--Efficient-Large-Model--Sana_1600M_512px_diffusers/snapshots/e58e81ec2dc4faf305122872313884df3afeffa8"


@dataclass(frozen=True)
class ExperimentSpec:
    scope: str
    learning_rate: float
    source_frame_stride: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    scheduler: str
    warmup_steps: int
    unfreeze_mllm: bool
    resume_from_checkpoint: str
    note: str
    min_source_frame_index: int = 0
    trajectory_motion_filter: bool = False
    trajectory_motion_start_threshold: float = 0.05
    trajectory_motion_start_padding: int = 0
    min_trajectory_delta: float = 0.0

    def key(self) -> str:
        return "|".join(
            [
                self.scope,
                self.note,
                f"lr={self.learning_rate:g}",
                f"stride={self.source_frame_stride}",
                f"motion={int(self.trajectory_motion_filter)}",
                f"mindelta={self.min_trajectory_delta:g}",
            ]
        )


def run(
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
    stdout=None,
    stderr=None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "failed": []}


def save_state(state: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def existing_fexp_numbers() -> list[int]:
    numbers: list[int] = []
    for root in (CONFIG_DIR, FINETUNE_SCRIPT_DIR):
        if not root.exists():
            continue
        for path in root.glob("F-exp*"):
            match = re.search(r"F-exp(\d+)", path.name)
            if match:
                numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def next_fexp_number() -> int:
    numbers = existing_fexp_numbers()
    return (max(numbers) + 1) if numbers else 1


def experiment_plan() -> list[ExperimentSpec]:
    """Motion-focused grid for Aloha future-frame training."""
    pretrained = "./foreact-pretrained"
    strong_prior = "./checkpoints/F-exp06-unfreeze-mllm-10epoch/checkpoint-17690"
    return [
        ExperimentSpec("banana", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "banana_motion_start_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5),
        ExperimentSpec("cube", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "cube_motion_start_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5),
        ExperimentSpec("mixed", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "mixed_motion_start_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5),
        ExperimentSpec("banana", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "banana_motion_delta005_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("cube", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "cube_motion_delta005_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("mixed", 1e-5, 5, 4, 4, 8.0, "constant_with_warmup", 0, False, pretrained, "mixed_motion_delta005_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("mixed", 3e-5, 5, 4, 4, 8.0, "cosine_with_min_lr", 100, False, pretrained, "mixed_motion_delta005_lr3e-5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("mixed", 5e-6, 3, 4, 4, 10.0, "cosine_with_min_lr", 100, False, pretrained, "mixed_motion_delta005_s3", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("banana", 3e-5, 3, 4, 4, 10.0, "cosine_with_min_lr", 100, False, pretrained, "banana_motion_delta005_s3_lr3e-5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("cube", 3e-5, 3, 4, 4, 10.0, "cosine_with_min_lr", 100, False, pretrained, "cube_motion_delta005_s3_lr3e-5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("mixed", 1e-5, 5, 2, 8, 8.0, "cosine_with_min_lr", 100, True, pretrained, "mixed_motion_unfreeze_s5", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
        ExperimentSpec("mixed", 5e-6, 5, 2, 8, 8.0, "cosine_with_min_lr", 100, True, strong_prior, "mixed_prior_motion_unfreeze_lr5e-6", trajectory_motion_filter=True, trajectory_motion_start_threshold=0.05, trajectory_motion_start_padding=5, min_trajectory_delta=0.05),
    ]


def scope_datasets(scope: str) -> dict[str, Path]:
    if scope == "banana":
        return {"aloha_banana": BANANA_DATASET}
    if scope == "cube":
        return {"aloha_cube": CUBE_DATASET}
    if scope == "mixed":
        return {"aloha_banana": BANANA_DATASET, "aloha_cube": CUBE_DATASET}
    raise ValueError(f"Unknown scope: {scope}")


def make_split_root(exp_name: str, scope: str) -> Path:
    root = SPLIT_ROOT / exp_name
    root.mkdir(parents=True, exist_ok=True)
    for name, target in scope_datasets(scope).items():
        link = root / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(target.resolve(), target_is_directory=True)
    return root


def config_for(exp_name: str, spec: ExperimentSpec, data_path: Path) -> dict:
    modules_to_freeze = ["vae"] if spec.unfreeze_mllm else ["vae", "mllm_backbone"]
    cfg = {
        "mllm_id": LOCAL_MLLM,
        "diffusion_model_id": LOCAL_SANA,
        "vae_id": LOCAL_SANA,
        "noise_scheduler_id": LOCAL_SANA,
        "scheduler_id": LOCAL_SANA,
        "vae_downsample_f": 32,
        "in_channels": 32,
        "system_prompt": "You are a robot and should focus on your actions. Generate a new image that meets the user's instruction while maintaining consistency with the original input where appropriate.",
        "_gradient_checkpointing": True,
        "modules_to_freeze": modules_to_freeze,
        "data_path": str(data_path.relative_to(PROJECT_ROOT)),
        "camera_key": "observation.images.cam_high",
        "target_image_size": [480, 640],
        "filtered_episodes_path": "",
        "cot_json_path": "",
        "subtask_data_path": "",
        "custom_data_path": "",
        "balance_datasets": False,
        "target_frame_offset": 30,
        "source_frame_stride": spec.source_frame_stride,
        "min_source_frame_index": spec.min_source_frame_index,
        "trajectory_motion_filter": spec.trajectory_motion_filter,
        "trajectory_key": "observation.state",
        "trajectory_motion_start_threshold": spec.trajectory_motion_start_threshold,
        "trajectory_motion_start_padding": spec.trajectory_motion_start_padding,
        "min_trajectory_delta": spec.min_trajectory_delta,
        "per_device_train_batch_size": spec.per_device_train_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": spec.learning_rate,
        "weight_decay": 0.05,
        "max_grad_norm": 0.5,
        "lr_scheduler_type": spec.scheduler,
        "warmup_steps": spec.warmup_steps,
        "num_train_epochs": spec.num_train_epochs,
        "save_strategy": "epoch",
        "save_total_limit": 4,
        "logging_steps": 5,
        "dataloader_num_workers": 12,
        "dataloader_persistent_workers": False,
        "dataloader_pin_memory": True,
        "bf16": True,
        "tf32": True,
        "deepspeed": "configs/zero1.json",
        "output_dir": "./checkpoints",
        "overwrite_output_dir": False,
        "resume_from_checkpoint": spec.resume_from_checkpoint,
        "run_name": exp_name,
        "report_to": "none",
    }
    if spec.scheduler == "cosine_with_min_lr":
        cfg["lr_scheduler_kwargs"] = {"min_lr": max(1e-6, spec.learning_rate / 10)}
    return cfg


def write_experiment_files(exp_num: int, spec: ExperimentSpec) -> tuple[str, Path, Path]:
    exp_name = f"F-exp{exp_num:02d}-aloha-{spec.scope}-{spec.note}"
    data_path = make_split_root(exp_name, spec.scope)
    config_path = CONFIG_DIR / f"{exp_name}.yaml"
    script_path = FINETUNE_SCRIPT_DIR / f"{exp_name}.sh"

    cfg = config_for(exp_name, spec, data_path)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {PROJECT_ROOT}",
                "export FORCE_VIDEO_BACKEND=pyav",
                "export WANDB_MODE=offline",
                'NUM_GPUS="${1:-4}"',
                f'conda run --no-capture-output -n foreact accelerate launch --num_processes "$NUM_GPUS" --main_process_port {26000 + exp_num} --mixed_precision bf16 train.py --config_file "{config_path.name}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return exp_name, config_path, script_path


def git_has_changes(paths: Iterable[Path]) -> bool:
    rels = [str(path.relative_to(PROJECT_ROOT)) for path in paths]
    status = run(["git", "status", "--porcelain", "--", *rels], check=True, stdout=subprocess.PIPE).stdout
    return bool(status.strip())


def commit_paths(message: str, paths: Iterable[Path]) -> None:
    path_list = list(paths)
    if not git_has_changes(path_list):
        return
    rels = [str(path.relative_to(PROJECT_ROOT)) for path in path_list]
    run(["git", "add", "--", *rels])
    run(["git", "commit", "-m", message])


def detect_num_gpus(default: int) -> int:
    try:
        proc = run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        count = len([line for line in proc.stdout.splitlines() if line.strip()])
        return count or default
    except Exception:
        return default


def gpu_compute_pids() -> list[int]:
    try:
        proc = run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return []
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def kill_gpu_jobs() -> None:
    pids = gpu_compute_pids()
    current = os.getpid()
    pids = [pid for pid in pids if pid != current]
    if not pids:
        return
    print(f"[runner] Killing existing GPU compute jobs: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def launch_training(exp_name: str, config_path: Path, num_gpus: int, conda_env: str, exp_num: int, cuda_devices: str | None = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{exp_name}.log"
    env = os.environ.copy()
    env["FORCE_VIDEO_BACKEND"] = "pyav"
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = env.get("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env["HF_HUB_ETAG_TIMEOUT"] = env.get("HF_HUB_ETAG_TIMEOUT", "30")
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "accelerate",
        "launch",
        "--num_processes",
        str(num_gpus),
        "--main_process_port",
        str(26000 + exp_num),
        "--mixed_precision",
        "bf16",
        "train.py",
        "--config_file",
        config_path.name,
    ]
    device_msg = f" CUDA_VISIBLE_DEVICES={cuda_devices}" if cuda_devices else ""
    print(f"[runner] Launching {exp_name} on {num_gpus} GPU(s).{device_msg} Log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launch{device_msg} {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def gpu_groups(num_gpus: int, parallel_jobs: int, gpus_per_job: int) -> list[str]:
    count = max(num_gpus, parallel_jobs * gpus_per_job)
    groups = []
    for i in range(parallel_jobs):
        start = i * gpus_per_job
        stop = min(start + gpus_per_job, count)
        groups.append(",".join(str(idx) for idx in range(start, stop)))
    return [group for group in groups if group]


def run_jobs_parallel(jobs: list[tuple[int, str, Path, str]], args, num_gpus: int) -> None:
    groups = gpu_groups(num_gpus, args.parallel_jobs, args.gpus_per_job)
    if not groups:
        return

    job_queue: queue.Queue[tuple[int, str, Path, str]] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    state_lock = threading.Lock()

    def worker(worker_idx: int, devices: str) -> None:
        while True:
            try:
                exp_num, exp_name, config_path, spec_key = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                code = launch_training(
                    exp_name,
                    config_path,
                    len(devices.split(",")),
                    args.conda_env,
                    exp_num,
                    cuda_devices=devices,
                )
                with state_lock:
                    state = load_state()
                    record = {"name": exp_name, "spec_key": spec_key, "time": time.time(), "worker": worker_idx, "cuda_visible_devices": devices}
                    if code == 0:
                        state.setdefault("completed", []).append(exp_name)
                        state.setdefault("completed_specs", []).append(spec_key)
                        state.setdefault("completed_records", []).append(record)
                    else:
                        record["exit_code"] = code
                        state.setdefault("failed", []).append(record)
                    save_state(state)
            finally:
                job_queue.task_done()
                time.sleep(10)

    threads = []
    for idx, devices in enumerate(groups):
        thread = threading.Thread(target=worker, args=(idx, devices), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def fallback_starvla(conda_env: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "starvla_act_cot.log"
    cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", str(STARVLA_SCRIPT)]
    cwd = STARVLA_SCRIPT.parents[2] if STARVLA_SCRIPT.is_file() else PROJECT_ROOT
    print(f"[runner] No ForeAct job available. Launching starVLA fallback. Log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} fallback {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="foreact")
    parser.add_argument("--fallback-conda-env", default="starVLA")
    parser.add_argument("--num-gpus", type=int, default=0, help="0 means auto-detect, falling back to --default-gpus.")
    parser.add_argument("--default-gpus", type=int, default=8)
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--gpus-per-job", type=int, default=4)
    parser.add_argument("--max-experiments", type=int, default=0, help="0 runs the full built-in plan.")
    parser.add_argument("--kill-existing-gpu-jobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--idle-restart-seconds", type=int, default=60)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    plan = experiment_plan()
    if args.max_experiments > 0:
        plan = plan[: args.max_experiments]

    if args.kill_existing_gpu_jobs:
        kill_gpu_jobs()

    jobs: list[tuple[int, str, Path, str]] = []
    completed_specs = set(state.get("completed_specs", []))
    for spec in plan:
        spec_key = spec.key()
        if spec_key in completed_specs:
            continue
        exp_num = next_fexp_number()
        exp_name, config_path, script_path = write_experiment_files(exp_num, spec)
        commit_paths(
            f"[{exp_name}] add aloha training config",
            [config_path, script_path],
        )
        jobs.append((exp_num, exp_name, config_path, spec_key))

    num_gpus = args.num_gpus if args.num_gpus > 0 else detect_num_gpus(args.default_gpus)
    if jobs:
        run_jobs_parallel(jobs, args, num_gpus)

    while True:
        code = fallback_starvla(args.fallback_conda_env)
        state = load_state()
        state.setdefault("fallback_runs", []).append({"exit_code": code, "time": time.time()})
        save_state(state)
        time.sleep(args.idle_restart_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
