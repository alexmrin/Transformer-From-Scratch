import sys
from typing import Optional
from pathlib import Path
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import wandb

from tools.config import TrainConfig
from tools.exceptions import ConfigException
from tools.torch_utils import (
    barrier,
    get_local_rank,
    get_world_size,
    get_global_rank,
    seed_all,
)
from tools.trainer import Trainer
from model.model import DecoderModel
from data.openwebtext.dataset import OpenWebTextDataset

def main(cfg: TrainConfig) -> None:
    if cfg.run_name is None:
        raise ConfigException("Run name must be assigned")

    barrier()

    # Assigning the device
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    cfg.model.precision = cfg.precision
    assert cfg.global_train_batch_size % get_world_size() == 0, "Batch size is incompatible with world size."
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size % cfg.device_train_microbatch_size == 0, f"Batch size is incompatible with microbatch size."
    cfg.device_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    assert cfg.global_eval_batch_size % get_world_size() == 0, "Eval batch size is inncompatible with world size."
    cfg.device_eval_batch_size = cfg.global_eval_batch_size // get_world_size()

    if get_global_rank() == 0:
        print("Configuration:")
        print(cfg)
    
    barrier()

    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            config=cfg.asdict(exclude=["wandb"]),
        )

    barrier()

    # Setting the seed
    seed_all(cfg.seed)

    class SequentialDistributedSampler(DistributedSampler):
        def __init__(self, dataset, num_replicas=None, rank=None):
            super().__init__(dataset, num_replicas, rank, shuffle=False)
        
        def __iter__(self):
            # Total number of samples in the dataset
            num_samples = math.ceil(len(self.dataset) / self.num_replicas)
            
            # Starting and ending indices for samples handled by this replica
            self.start_index = self.rank * num_samples
            self.end_index = min(self.start_index + num_samples, len(self.dataset))
            
            return iter(range(self.start_index, self.end_index))

        def __len__(self):
            return self.end_index - self.start_index
    
    train_dataset = OpenWebTextDataset(cfg.data.train_pth, chunk_size=cfg.model.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.device_train_batch_size,
        sampler=SequentialDistributedSampler(train_dataset),
        pin_memory=cfg.data.pin_memory,
        num_workers=cfg.data.num_workers
    )
    val_dataset = OpenWebTextDataset(cfg.data.val_pth, chunk_size=cfg.model.max_seq_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.device_eval_batch_size,
        sampler=SequentialDistributedSampler(val_dataset),
        pin_memory=cfg.data.pin_memory,
        num_workers=cfg.data.num_workers
    )

    barrier()

    # init model
    print("Building model...")
    model = DecoderModel(cfg.model)
    print(f"Total number of parameters: {model.num_params():,d}")
    print(f"Number of non-embedding parameters: {model.num_params(include_embedding=False):,d}")

    model.to(device)

    # TODO: Add more customization with optimizer and lr_scheduler
    optimizer = model.configure_optimizers(cfg.optimizer.weight_decay, cfg.optimizer.learning_rate, cfg.optimizer.betas, "cuda")
    
    total_steps = cfg.max_steps
    def warmup_scheduler(step, total_steps):
        warmup_steps = int(total_steps * cfg.optimizer.warmup_prop)
        if step < warmup_steps:
            return cfg.optimizer.min_lr / cfg.optimizer.learning_rate + (1 - cfg.optimizer.min_lr / cfg.optimizer.learning_rate) * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return ((cfg.optimizer.learning_rate - cfg.optimizer.min_lr) * cosine_decay + cfg.optimizer.min_lr) / cfg.optimizer.learning_rate
        
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_scheduler(step, total_steps))

    if cfg.resume_from_ckpt:
        print(f"Resuming training from checkpoint: {os.path.join(cfg.save_folder, 'ckpt.pt')}")
        checkpoint = torch.load(os.path.join(cfg.save_folder, 'ckpt.pt'))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
    else:
        step = 0

    # Requires Pytorch 2.0
    if cfg.compile:
        print("Compiling the model...")
        model = torch.compile(model)

    model = DDP(model, device_ids=[get_local_rank()])

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        step=step
    )
    if not cfg.dry_run:
        print("Starting training...")
        if get_global_rank() == 0 and cfg.wandb is not None:
            wandb.watch(model, log="all", log_freq=cfg.log_interval)
        trainer.fit()
        print("Training complete!")
    else:
        print("Dry run complete!")

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    cfg = TrainConfig()
    main(cfg)
    dist.destroy_process_group()