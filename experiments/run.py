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
import pytorch_lightning as pl
import torch.nn.functional as F
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from pseudo_labeller.model import PsuedoIrradienceForecastor


class LitModel(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.forecast_steps = config.forecast_steps
        self.learning_rate = config.lr
        self.model = PsuedoIrradienceForecastor(
            input_channels=config.input_channels,
            input_size=config.input_size,
            input_steps=config.input_steps,
            output_channels=config.latent_channels,
            conv3d_channels=config.conv3d_channels,
            hidden_dim=config.hidden_dim,
            kernel_size=config.kernel_size,
            num_layers=config.num_layers,
            output_steps=config.output_steps,
            pv_meta_input_channels=config.pv_meta_input_channels,
        )
        self.config = self.model.config
        self.save_hyperparameters()

    def forward(self, x, meta):
        return self.model(x, meta, output_latents=False)

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, meta, y = batch
        x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
        y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
        x = x.half()
        y = y.half()
        meta = meta.half()
        y_hat = self(x, meta)

        mask = meta > 0.0

        # Expand to match the ground truth shape
        mask = einops.repeat(mask, "b c h w -> b c t h w", t=y.shape[2])

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat[mask], y[mask])
        nmae_loss = (y_hat[mask] - y[mask]).abs().mean()
        loss = nmae_loss
        self.log("loss", loss)
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def create_train_dataloader(config: DictConfig):
    return pseudo_irradiance_datapipe(
        config.config,
        start_time=datetime.datetime(2008, 1, 1),
        end_time=datetime.datetime(2020, 12, 31),
        use_sun=config.sun,
        use_nwp=config.nwp,
        use_sat=config.sat,
        use_hrv=config.hrv,
        use_pv=True,
        use_topo=config.topo,
        size=config.size,
        use_future=config.use_future,
    )


def create_val_dataloader(config: DictConfig):
    return pseudo_irradiance_datapipe(
        config.config,
        start_time=datetime.datetime(2021, 1, 1),
        end_time=datetime.datetime(2021, 12, 31),
        use_sun=config.sun,
        use_nwp=config.nwp,
        use_sat=config.sat,
        use_hrv=config.hrv,
        use_pv=True,
        use_topo=config.topo,
        size=config.size,
        use_future=config.use_future,
    )


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
