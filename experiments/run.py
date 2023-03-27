"""Code for running experiments"""
import torch

try:
    torch.multiprocessing.set_start_method("spawn")
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
except RuntimeError:
    pass

import hydra
from omegaconf import DictConfig, OmegaConf
from .irradiance.model import LitIrradianceModel
from .site_level.model import LitMetNetModel
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(version_base=None, config_path="configs", config_name="config")
def experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.model_name == "metnet":
        model = LitMetNetModel(cfg.model, dataloader_config=cfg.dataloader)
    elif cfg.model_name == "irradiance":
        model = LitIrradianceModel(cfg.model, dataloader_config=cfg.dataloader)
    else:
        raise ValueError(f"Unknown model name {cfg.model_name}")

    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        monitor="step",
        mode="max",
        save_last=True,
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=cfg.epochs,
        precision=16 if cfg.fp16 else 32,
        devices="auto" if not cfg.cpu else "cpu",
        accelerator="auto" if not cfg.cpu else "cpu",
        log_every_n_steps=1,
        # limit_val_batches=400 * args.accumulate,
        # limit_train_batches=500 * args.accumulate,
        accumulate_grad_batches=cfg.accumulate,
        callbacks=[model_checkpoint, lr_monitor],
        logger=[TensorBoardLogger(save_dir="logs/"),
                WandbLogger(project="PvMetNet" if cfg.model_name == "metnet" else "PvIrradiance",
                            log_model="all",)]
    )
    trainer.fit(model)

if __name__ == "__main__":
    experiment()
