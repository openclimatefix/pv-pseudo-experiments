import pytorch_lightning as pl
import torch.nn.functional as F

from pseudo_labeller.model import PsuedoIrradienceForecastor
import torch
import einops

import datetime

import einops
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig


class LitIrradianceModel(pl.LightningModule):
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