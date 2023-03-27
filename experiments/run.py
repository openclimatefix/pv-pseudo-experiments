"""Code for running experiments"""
import torch

try:
    torch.multiprocessing.set_start_method("spawn")
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
except RuntimeError:
    pass
import datetime

import einops
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig, OmegaConf





@hydra.main(version_base=None, config_path="configs", config_name="config")
def experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = LitModel(cfg.model)
    train_dataloader = create_train_dataloader(cfg.dataloader)
    val_dataloader = create_val_dataloader(cfg.dataloader)

    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        monitor="step",
        mode="max",
        save_last=True,
        save_top_k=10,
    )

    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    # early_stopping = EarlyStopping(monitor="loss")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        precision=16 if cfg.fp16 else 32,
        devices=[cfg.num_gpu] if not cfg.cpu else 1,
        accelerator="auto" if not cfg.cpu else "cpu",
        log_every_n_steps=1,
        # limit_val_batches=400 * args.accumulate,
        # limit_train_batches=500 * args.accumulate,
        accumulate_grad_batches=cfg.accumulate,
        callbacks=[model_checkpoint],
        logger=tb_logger,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    experiment()
