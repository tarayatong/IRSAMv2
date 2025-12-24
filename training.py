import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from sam_spl.base_model import make_adaptor
from training.Adan import Adan
from dataset.image_floder import ImageFolder
from training import Trainer, seed_everything, metricWrapper
from loguru import logger
import datetime
import yaml


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="SAM-SPL distributed training script supporting multiple datasets and distributed training configuration."
    )
    parser.add_argument(
        "--dataset",
        default="NUDT-SIRST",
        type=str,
        help="Dataset name, e.g., NUDT-SIRST, IRSTDID=SKY, IRSTD-1k, etc.",
    )
    parser.add_argument("--image_size", default=256, type=int, help="Input image size")
    parser.add_argument(
        "--batch_size", default=12, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--epoch", default=300, type=int, help="Number of training epochs"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument(
        "--save_dir",
        default="./checkpoints/IRSTD-1k",
        type=str,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--data_path",
        default="./dataset/set_configs",
        type=str,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--config_path",
        default="./sam_spl/configs/spl_t.yaml",
        type=str,
        help="Path to config file for model adaptor",
    )
    parser.add_argument(
        "--loss_func",
        default="bceloss",
        type=str,
        help="Loss function to use (default: bceloss)",
    )
    parser.add_argument(
        "--eta_min",
        default=1e-6,
        type=float,
        help="Minimum learning rate for scheduler",
    )
    parser.add_argument("--seed", default=2727, type=int, help="Random seed")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Enable DistributedDataParallel (DDP) training",
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU id to use (if not using DDP)"
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    distributed = args.use_ddp
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))


    # Set up logging (only main process logs in DDP)
    is_main_process = (not distributed) or (local_rank == 0)
    if is_main_process:
        os.makedirs("./logs", exist_ok=True)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"./logs/training_{args.dataset}_{now_str}.log"
        logger.add(
            log_filename,
            format="{time} - {level} - {message}",
            encoding="utf-8",
            rotation="10 MB",
        )
        # Log argument parser description and arguments
        logger.info(f"ArgumentParser description: {parser.description}")
        arg_items = list(vars(args).items())
        max_key_len = max(len(str(k)) for k, _ in arg_items)
        logger.info("Arguments table:\n" +
            "\n".join([
                f"| {'Argument'.ljust(max_key_len)} | Value           |",
                f"|{'-' * (max_key_len+2)}|------------------|"
            ] + [
                f"| {str(k).ljust(max_key_len)} | {str(v)}" for k, v in arg_items
            ]
        ))

    
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    train_set = ImageFolder(
        args.data_path,
        args.dataset,
        istraining=True,
        base_size=args.image_size,
        crop_size=args.image_size,
    )
    val_set = ImageFolder(
        args.data_path,
        args.dataset,
        istraining=False,
        base_size=args.image_size,
        crop_size=args.image_size,
    )

    # Load model configuration from YAML file
    with open(args.config_path, "r") as f:
        adaptor_cfg = yaml.safe_load(f)
    model = make_adaptor(**adaptor_cfg)
    model._freeze_encoder()

    # Set up optimizer, scheduler, loss function
    optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=args.eta_min
    )
    loss_fun = nn.BCELoss(reduction="mean")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_set,
        val_dataset=val_set,
        loss_fn=loss_fun,
        device=device,
        batch_size=args.batch_size,
        num_workers=7,
        distributed=distributed,
        save_dir=args.save_dir,
        metric_wrapper = metricWrapper(),
    )
    

    # Training loop
    for epoch in range(args.epoch):
        trainer.epoch = epoch
        trainer.train_one_epoch()
        if epoch % 1 == 0:
            val_loss, val_metrics = trainer.evaluate()
            trainer.save_checkpoint(
                save_path=os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt")
            )


if __name__ == "__main__":
    main()
