"""
Trainer Module for SAM-SPL Model Training

This module provides a comprehensive training framework for the SAM-SPL model.
It supports both single-GPU and distributed training, with features for
training, validation, checkpointing, and metric evaluation.

Key Features:
- Distributed training support with PyTorch DDP
- Progress tracking with enlighten progress bars
- Automatic checkpoint saving and loading
- Flexible metric evaluation system
- Support for custom loss functions
"""

import os
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import enlighten

# Import metricWrapper for type hints
from .metrics_config import metricWrapper


class Trainer:
    """
    A comprehensive training class for SAM-SPL models.
    
    This class handles the complete training lifecycle including:
    - Model training and validation
    - Distributed training setup
    - Checkpoint management
    - Progress monitoring
    - Metric evaluation
    
    Attributes:
        model: The neural network model to train
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler (optional)
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        loss_fn: Custom loss function (default: BCELoss)
        device: Device to run training on (default: "cuda")
        batch_size: Batch size for training (default: 8)
        num_workers: Number of data loading workers (default: 4)
        distributed: Enable distributed training (default: False)
        save_dir: Directory to save checkpoints (default: "./checkpoints")
        etric_wrapper: Optional metric computation wrapper
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: torch.utils.data.Dataset,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        distributed: bool = False,
        save_dir: str = "./checkpoints",
        metric_wrapper: Optional[metricWrapper] = metricWrapper(),
    ):
        self.model = model  # type: nn.Module
        self.optimizer = optimizer  # type: torch.optim.Optimizer
        self.scheduler = scheduler  # type: Optional[torch.optim.lr_scheduler.LRScheduler]
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss(reduction="mean")  # type: nn.Module
        self.device = device  # type: Union[str, torch.device]
        self.save_dir = save_dir  # type: str
        self.metric_wrapper = metric_wrapper  # type: Optional[metricWrapper]
        os.makedirs(self.save_dir, exist_ok=True)
        self.distributed = distributed  # type: bool
        self.rank = dist.get_rank() if distributed else 0  # type: int
        self.world_size = dist.get_world_size() if distributed else 1  # type: int
        self.epoch = 0  # type: int
        self.best_metric = None  # type: Optional[Any]
        self.train_sampler = (
            DistributedSampler(train_dataset) if distributed else None
        )  # type: Optional[DistributedSampler]
        self.val_sampler = (
            DistributedSampler(val_dataset)
            if (distributed and val_dataset is not None)
            else None
        )  # type: Optional[DistributedSampler]
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
        )  # type: DataLoader
        self.val_loader = None  # type: Optional[DataLoader]
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=self.val_sampler,
            )  # type: DataLoader
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device] if torch.cuda.is_available() else None,
                output_device=self.rank,
                find_unused_parameters=True,
            )

    def train_one_epoch(self) -> float:
        """
        Train the model for one complete epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        total_loss = 0
        total_samples = 0
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader),
                desc=f"[Epoch {self.epoch: 3d}] Training",
                unit="batch",
                color="green",
                leave=False,
            )
        for idx, batch in enumerate(self.train_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logits = self.model(batch_data)
            loss = 0
            for pred_logit in pred_logits:
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Training] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
        if self.rank == 0:
            pbar.close()
        avg_loss = total_loss / total_samples
        if self.scheduler is not None:
            self.scheduler.step()
        if self.rank == 0:
            logger.info(f"[Train][Epoch {self.epoch}] avg_loss={avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Optional[Tuple[float, Optional[metricWrapper]]]:
        """
        Evaluate the model on the validation dataset.
            
        Returns:
            tuple: (average_loss, metric_wrapper) or None if no validation data
        """
        if self.metric_wrapper is not None:
            self.metric_wrapper.reset()
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.val_loader),
                desc=f"[Epoch {self.epoch: 3d}] Evaluating",
                unit="batch",
                color="red",
                leave=False,
            )
        self.model.eval()
        if self.val_loader is None:
            return None
        total_loss = 0
        total_samples = 0
        for idx, batch in enumerate(self.val_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logit = self.model(batch_data)[0]
            loss = 0
            loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Evaluating] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
            if self.metric_wrapper:
                self.metric_wrapper(pred_logit, batch_masks)
        avg_loss = total_loss / total_samples
        if self.rank == 0:
            logger.info(
                f"[Eval][Epoch {self.epoch}] avg_loss={avg_loss:.6f} metrics: {self.metric_wrapper}"
            )
        return avg_loss, self.metric_wrapper

    def save_checkpoint(self, save_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint to disk.
        
        Only rank 0 process saves checkpoints in distributed training.
        
        Args:
            save_path: Path to save the checkpoint
            extra: Additional data to include in checkpoint (optional)
        """
        if self.rank != 0:
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.module.state_dict()
            if self.distributed
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
        }
        if extra:
            state.update(extra)
        torch.save(state, save_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint from disk.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            dict: Loaded checkpoint dictionary
        """
        map_location = (
            {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.distributed else self.device
        )
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", None)
        return checkpoint
