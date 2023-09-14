"""Code for running experiments"""
import torch

try:
    torch.multiprocessing.set_start_method("spawn")
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
except RuntimeError:
    pass

import hydra
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from pv_pseudo_experiments.irradiance.model import LitIrradianceModel
from pv_pseudo_experiments.irradiance.batch_writing import BatchWriter
from pv_pseudo_experiments.site_level.batch_writing import PVBatchWriter
from pv_pseudo_experiments.site_level.model import LitMetNetModel
from pv_pseudo_experiments.irradiance.model_inference import LitIrradianceInferenceModel
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path="configs", config_name="batch")
def experiment(cfg: DictConfig) -> None:
    cfg.model.input_size = cfg.dataloader.size
    print(OmegaConf.to_yaml(cfg))
    #print("gpu id:", HydraConfig.get().job.num % cfg.num_gpus)
    if cfg.model_name == "metnet":
        model = LitMetNetModel(cfg.model, dataloader_config=cfg.dataloader)
    elif cfg.model_name == "irradiance":
        model = LitIrradianceModel(cfg.model, dataloader_config=cfg.dataloader)
    elif cfg.model_name == "irradiance_inference":
        model = LitIrradianceInferenceModel(cfg.model, dataloader_config=cfg.dataloader)
        trainer = Trainer(
            max_epochs=cfg.epochs,
            precision=16 if cfg.fp16 else 32,
            devices=[3],
            accelerator="gpu",
            log_every_n_steps=1,
            # limit_val_batches=400 * args.accumulate,
            # limit_train_batches=500 * args.accumulate,
            accumulate_grad_batches=cfg.accumulate,
        )
        trainer.test(model)
        exit()
    elif cfg.model_name == "batch":
        model = BatchWriter(cfg.dataloader)
        # Create batches now
        model()
        exit()
    elif cfg.model_name == "metnet_batch":
        model = PVBatchWriter(cfg.dataloader)
        # Create batches now
        model()
        exit()
    else:
        raise ValueError(f"Unknown model name {cfg.model_name}")

    model_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    loggers = []
    if cfg.wandb:
        loggers.append(WandbLogger(project="PvMetNet" if cfg.model_name == "metnet" else "PvIrradiance",
                            log_model="all",save_dir="/mnt/storage_ssd_4tb/wandb", name=f"input{cfg.model.input_size}"
                                                  f"_output{cfg.model.output_steps}"
                                                  f"_conv3d{cfg.model.conv3d_channels}"
                                                  f"_latent{cfg.model.latent_channels}"
                                                  f"_layers{cfg.model.num_layers}"
                                                  f"_hidden{cfg.model.hidden_dim}"
                                                  f"_sun{cfg.dataloader.sun}"
                                                  f"_nwp{cfg.dataloader.nwp}"
                                                  f"_sat{cfg.dataloader.sat}"
                                                  f"_hrv{cfg.dataloader.hrv}"
                                                  f"_topo{cfg.dataloader.topo}"
                                                  f"_batch{cfg.dataloader.batch}"
                                                  f"_fp16{cfg.fp16}"
                                                  f"_lr{cfg.model.lr}",))
        loggers[-1].watch(model, log="all")
    trainer = Trainer(
        max_epochs=cfg.epochs,
        precision=16 if cfg.fp16 else 32,
        devices=[HydraConfig.get().job.num % cfg.num_gpus] if not cfg.cpu else "cpu",
        accelerator="auto" if not cfg.cpu else "cpu",
        log_every_n_steps=500,
        # limit_val_batches=400 * args.accumulate,
        # limit_train_batches=500 * args.accumulate,
        accumulate_grad_batches=cfg.accumulate,
        callbacks=[model_checkpoint, lr_monitor],
        logger=loggers
    )
    trainer.fit(model)

if __name__ == "__main__":
    experiment()

