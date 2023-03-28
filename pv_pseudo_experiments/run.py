"""Code for running experiments"""
import torch

try:
    torch.multiprocessing.set_start_method("spawn")
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
except RuntimeError:
    pass

import hydra
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from pv_pseudo_experiments.irradiance.model import LitIrradianceModel
from pv_pseudo_experiments.site_level.model import LitMetNetModel


@hydra.main(version_base=None, config_path="configs", config_name="pv_metnet")
def experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.model_name == "metnet":
        model = LitMetNetModel(cfg.model, dataloader_config=cfg.dataloader)
    elif cfg.model_name == "irradiance":
        model = LitIrradianceModel(cfg.model, dataloader_config=cfg.dataloader)
    else:
        raise ValueError(f"Unknown model name {cfg.model_name}")

    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=500,
        monitor="step",
        mode="max",
        save_last=True,
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    loggers = [TensorBoardLogger(save_dir="./")]
    if cfg.wandb:
        loggers.append(WandbLogger(project="PvMetNet" if cfg.model_name == "metnet" else "PvIrradiance",
                            log_model="all",))
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
        logger=loggers
    )
    trainer.fit(model)

if __name__ == "__main__":
    experiment()
