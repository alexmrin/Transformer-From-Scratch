from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, List, Tuple, Iterable, Any, Dict
import json

import torch

@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    n_kvheads: int = 4
    d_intermediate: int = 3072
    p_dropout: float = 0
    vocab_size: int = 50258
    eps: float = 1e-6
    max_seq_len: int = 1024
    precision: Optional[Literal["amp_fp16", "amp_bf16", "fp32"]] = None # Set by the training config, do not explicitly assign

@dataclass
class DataConfig:
    train_pth: str = "./data/openwebtext/train.bin"
    val_pth: str = "./data/openwebtext/val.bin"
    num_proc: int = 16 # ~half of your cpus
    num_proc_load_dataset: int = 16 # ~half of your cpus
    total_batches: int = 1024
    pin_memory: bool = True
    num_workers: int = 12

@dataclass
class WandbConfig:
    project: Optional[str] = "GPT-transformer"
    group: Optional[str] = "OpenWebText"
    name: Optional[str] = "8xA100run2"
    log_artifacts: bool = True
    rank_zero_only: bool = True
    log_interval: int = 100

@dataclass
class OptimizerConfig:
    learning_rate: float = 6.0e-4
    weight_decay: float = 1.0e-1
    betas: Tuple[float, float] = (0.9, 0.95)
    min_lr: float = 6.0e-5
    warmup_prop: float = 0.05

@dataclass
class TrainConfig:
    max_steps: int = 200000
    run_name: Optional[str] = "test"
    seed: int = 908
    epoch: Optional[int] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    precision: Optional[Literal["amp_fp16", "amp_bf16", "fp32"]] = "amp_bf16"
    global_train_batch_size: int = 2048
    device_train_batch_size: Optional[int] = None # Set by global_train_batch_size // get_world_rank(), do not explicitly assign
    device_train_microbatch_size: int = 32 # Set as high as possible
    device_grad_accum: Optional[int] = None # Set by device_train_batch_size // device_train_microbatch_size, do not explicity assign
    global_eval_batch_size: int = 128
    device_eval_batch_size: Optional[int] = None # Do not assign
    wandb: WandbConfig = field(default_factory=WandbConfig)
    save_folder: str = "./saves"
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dry_run: bool = False
    compile: bool = False
    grad_clip: float = 4.0
    time_limit: Optional[float] = 60 * 60 * 47.5 # 48 hours
    log_interval: Optional[int] = 1000
    resume_from_ckpt: bool = False
    

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def __str__(self):
        return json.dumps(asdict(self), indent=4)