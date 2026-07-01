import datasets
import os
import shutil
import torch
import transformers
import yaml
import PIL.Image

from accelerate.utils import release_memory
from dataclasses import dataclass, field
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms.functional import to_pil_image
from PIL import PngImagePlugin

from dataloaders.dataset_finetune import get_train_datasets
from models.visualforesight import VisualForesightConfig, VisualForesight
from utils.trainer_utils import find_newest_checkpoint, possible_override_args, ModelCallback

datasets.disable_caching()
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "VisualForesight"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


@dataclass
class OverrideArguments:
    config_file: str = None


@dataclass
class ModelArguments:
    mllm_id: str = "google/gemma-2-2b-it"
    diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    max_input_text_tokens: int = 256
    vae_downsample_f: int = 32
    in_channels: int = 32
    system_prompt: str = "You are a robot and should focus on your actions. Generate a new image that meets the user's instruction while maintaining consistency with the original input where appropriate."
    _gradient_checkpointing: bool = True
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()


@dataclass
class DataArguments:
    data_path: str = "data/realworld"
    camera_key: str = "observation.images.head_left_rgb"
    target_image_size: tuple[int, int] = (480, 640)
    filtered_episodes_path: str = ""
    cot_json_path: str = ""
    subtask_data_path: str = ""
    target_frame_offset: int = 0  # >0: fixed offset (e.g. 6 = predict 6 frames ahead); 0: use subtask/cot logic
    source_frame_stride: int = 0  # >0: sample fixed-offset source frames every N frames; 0: use dataset fps
    min_source_frame_index: int = 0  # skip source frames before this absolute frame index
    clamp_target_frame_to_last: bool = False  # include tail source frames by clamping target to episode's last frame
    trajectory_motion_filter: bool = False  # skip each episode's low-motion prefix using parquet trajectory
    trajectory_key: str = "observation.state"
    trajectory_motion_start_threshold: float = 0.05
    trajectory_motion_start_padding: int = 0
    min_trajectory_delta: float = 0.0  # skip pairs whose trajectory change from source to target is too small
    custom_data_path: str = ""  # path to custom dataset dir (source/target image pairs + captions.json)
    balance_datasets: bool = False  # when True, use BalancedConcatDataset for 1:1 ratio
    frame_cache_root: str = ""  # optional decoded frame cache root, organized by dataset/camera/episode
    frame_cache_ext: str = "jpg"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "output"
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    warmup_steps: int = 5000
    logging_steps: int = 1
    save_steps: int = 1000
    save_total_limit: int = 1000
    restore_callback_states_from_checkpoint: bool = True
    save_final_checkpoint: bool = False
    seed: int = 42
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 4
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None

    def __post_init__(self):
        try:
            self = possible_override_args(override_args, self)
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Failed to load override config: {exc}")
        super().__post_init__()



class BalancedEpochCallback(TrainerCallback):
    """Notify the train dataset (e.g. BalancedConcatDataset) at the start of each epoch
    so it can regenerate its per-epoch sampling schedule."""

    def __init__(self, train_dataset):
        self._train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        ds = self._train_dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(int(state.epoch) if state.epoch is not None else 0)


class DebugTrainer(Trainer):
    """Trainer subclass that dumps the first batch (images + text) for debugging."""

    def __init__(self, *args, debug_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_tokenizer = debug_tokenizer
        self._debug_done = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._debug_done and self.args.process_index == 0:
            self._debug_done = True
            debug_dir = os.path.join(self.args.output_dir, "debug_batch")
            os.makedirs(debug_dir, exist_ok=True)

            print("\n" + "=" * 60)
            print("[DEBUG] Step-1 batch inspection")
            print(f"  Keys          : {list(inputs.keys())}")
            print(f"  source  shape : {inputs['source'].shape}  dtype={inputs['source'].dtype}")
            print(f"  target  shape : {inputs['target'].shape}  dtype={inputs['target'].dtype}")
            print(f"  input_ids     : {inputs['input_ids'].shape}")
            print(f"  attention_mask: {inputs['attention_mask'].shape}")

            # ---- decode text captions ----------------------------------------
            n_show = min(8, inputs["input_ids"].shape[0])
            print(f"\n  [Captions (first {n_show} samples)]")
            for i in range(n_show):
                ids = inputs["input_ids"][i]
                text = self._debug_tokenizer.decode(ids, skip_special_tokens=True)
                # strip padding / long system-prompt prefix for readability
                text = text.strip()
                print(f"    [{i}] {repr(text)}")

            # ---- save images -------------------------------------------------
            # images are normalised to [-1, 1]; denormalise back to [0, 1]
            src = (inputs["source"].cpu().float() * 0.5 + 0.5).clamp(0, 1)
            tgt = (inputs["target"].cpu().float() * 0.5 + 0.5).clamp(0, 1)
            for i in range(min(8, src.shape[0])):
                to_pil_image(src[i]).save(os.path.join(debug_dir, f"sample_{i:02d}_source.png"))
                to_pil_image(tgt[i]).save(os.path.join(debug_dir, f"sample_{i:02d}_target.png"))

            print(f"\n  [Images saved to {debug_dir}]")
            print("=" * 60 + "\n")

        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs)


if __name__ == "__main__":
    override_parser = transformers.HfArgumentParser((OverrideArguments))
    override_args = override_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[0]
    parser = transformers.HfArgumentParser(
        (OverrideArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    _, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = possible_override_args(override_args, model_args, data_args)

    assert (
        data_args.target_image_size[0] % model_args.vae_downsample_f == 0 and data_args.target_image_size[1] % model_args.vae_downsample_f == 0
    ), f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = (data_args.target_image_size[0] // model_args.vae_downsample_f, data_args.target_image_size[1] // model_args.vae_downsample_f)

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(
            training_args.resume_from_checkpoint
        )
        model = VisualForesight.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True,
            **model_args.__dict__,
        )
    else:
        model = VisualForesight(
            config=VisualForesightConfig(
                input_size=input_size,
                **model_args.__dict__,
            ),
        )

    with training_args.main_process_first(local=False):
        train_dataset, collate_fn = get_train_datasets(
            data_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
            base_seed=training_args.seed,
        )

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[ModelCallback(), BalancedEpochCallback(train_dataset)],
        debug_tokenizer=model.get_tokenizer(),
    )

    training_args.output_dir = str(
        os.path.join(training_args.output_dir, training_args.run_name)
    )
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(
            training_args.output_dir
        ):
            shutil.rmtree(training_args.output_dir)
        print(f"Training dataset size: {len(train_dataset)}")

    while (
        trainer.state.epoch is None
        or (training_args.num_train_epochs - trainer.state.epoch) > 0.01
    ):
        if trainer.state.epoch is not None:
            trainer.control.should_training_stop = False
            trainer.args.eval_on_start = False
            trainer.model = model
            (trainer.model_wrapped,) = release_memory(trainer.model_wrapped)
            trainer.model_wrapped = trainer.model
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)

        trainer.train(resume_from_checkpoint=last_checkpoint)

    if training_args.save_final_checkpoint:
        final_checkpoint_dir = os.path.join(
            training_args.output_dir, f"checkpoint-{trainer.state.global_step}"
        )
        trainer.save_model(final_checkpoint_dir)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(
                os.path.join(final_checkpoint_dir, "trainer_state.json")
            )
            print(f"Saved final checkpoint to {final_checkpoint_dir}")
