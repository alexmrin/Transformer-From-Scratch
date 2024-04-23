import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import wandb
from tqdm import tqdm

from tools.config import TrainConfig
from tools.torch_utils import get_global_rank

class Trainer():
    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.parallel.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        step = 0
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.is_master = get_global_rank() == 0
        self.step = step
        self.session_step = 0

    def eval(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_items = 0
        perplexity = 0.0
        num_batches = 20

        with torch.inference_mode():
            for i, (inputs, targets) in enumerate(self.val_loader):
                if i > num_batches:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                logits, loss = self.model(inputs, targets)
                total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
                total_items += inputs.size(0)

        # Calculate average loss and perplexity over all validation data
        average_loss = total_loss / total_items
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        self.model.train()
        return average_loss, perplexity
    
    def fit(self):
        scaler = GradScaler(enabled=(not self.cfg.precision == "fp32"))
        done = False
        start_time = time.time()

        while not done:
            for inputs, targets in self.train_loader:
                step_start = time.time()
                accumulated_loss = 0

                self.model.train()
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                for micro_step in range(self.cfg.device_grad_accum):
                    if self.step >= self.cfg.max_steps or (time.time() - start_time) >= self.cfg.time_limit:
                        done = True
                        break

                    self.model.require_backward_grad_sync = (micro_step == self.cfg.device_grad_accum - 1)

                    with torch.cuda.amp.autocast(enabled=(not self.cfg.precision == "fp32"), dtype=self.cfg.autocast_precision):
                        minibatch_inputs = inputs[micro_step*self.cfg.device_train_microbatch_size:(micro_step+1)*self.cfg.device_train_microbatch_size]
                        minibatch_targets = targets[micro_step*self.cfg.device_train_microbatch_size:(micro_step+1)*self.cfg.device_train_microbatch_size]
                        logits, loss = self.model(
                            minibatch_inputs,
                            minibatch_targets
                        )
                        if torch.isnan(loss):
                            print("NaN loss detected, stopping training.")
                            print(f"microbatch inputs: {minibatch_inputs}\nmicrobatch targets: {minibatch_targets}")
                            print(f"inputs: {inputs.shape}\ntargets: {targets.shape}")
                            print(f"microstep: {micro_step}")
                            done=True
                            break  # Break out of the inner loop
                        loss = loss / self.cfg.device_grad_accum
                    scaler.scale(loss).backward()
                    accumulated_loss += loss.item()
                if self.cfg.grad_clip != 0.0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                average_loss = accumulated_loss / self.cfg.device_train_batch_size
                step_time = time.time() - step_start

                if self.step % self.cfg.log_interval == 0 and self.step != 0:
                    eval_loss, eval_perplexity = self.eval()
                    if self.is_master:
                        wandb.log({
                            "training_loss": average_loss,
                            "step_time": step_time,
                            "eval_loss": eval_loss,
                            "eval_perplexity": eval_perplexity,
                            "step": self.step,
                            "learning_rate": self.optimizer.param_groups[0]['lr']
                        })
                        checkpoint = {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "step": self.step,
                            "config": self.cfg,
                        }
                        print(f"saving checkpoint to {self.cfg.save_folder}")
                        torch.save(checkpoint, os.path.join(self.cfg.save_folder, 'ckpt.pt'))
                        artifact = wandb.Artifact(f"model-checkpoint-{self.step}", type="model")
                        artifact.add_file(os.path.join(self.cfg.save_folder, 'ckpt.pt'))
                        wandb.log_artifact(artifact)
                else:
                    if self.is_master:
                        wandb.log({
                            "training_loss": average_loss,
                            "step_time": step_time,
                            "step": self.step,
                            "learning_rate": self.optimizer.param_groups[0]['lr']
                        })

                self.step += 1
                self.session_step += 1
                if self.is_master:
                    print(f"step number: {self.step}, session step number: {self.session_step}")
                if done:
                    break
        if self.is_master:
            wandb.finish()