import datetime

import einops
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig
from pseudo_labeller.model import PsuedoIrradienceForecastor
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torch.utils.data.dataloader import DataLoader

class LitIrradianceModel(LightningModule):
    def __init__(
            self,
            config: DictConfig,
            dataloader_config: DictConfig,
    ):
        super().__init__()
        self.forecast_steps = config.forecast_steps
        self.dataloader_config = dataloader_config
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
        #x = x.half()
        #y = y.half()
        #meta = meta.half()
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

    def train_dataloader(self):
        # Return your dataloader for training
        datapipe = pseudo_irradiance_datapipe(
            self.dataloader_config.config,
            start_time=datetime.datetime(2008, 1, 1),
            end_time=datetime.datetime(2020, 12, 31),
            use_sun=self.dataloader_config.sun,
            use_nwp=self.dataloader_config.nwp,
            use_sat=self.dataloader_config.sat,
            use_hrv=self.dataloader_config.hrv,
            use_pv=True,
            use_topo=self.dataloader_config.topo,
            size=self.dataloader_config.size,
            use_future=self.dataloader_config.use_future,
            batch_size=self.dataloader_config.batch,
        )
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(datapipe.collate().set_length(10000),
                          num_workers=self.dataloader_config.num_workers, batch_size=None)

    def test_dataloader(self):
        datapipe = pseudo_irradiance_datapipe(
            self.dataloader_config.config,
            start_time=datetime.datetime(2021, 1, 1),
            end_time=datetime.datetime(2021, 12, 31),
            use_sun=self.dataloader_config.sun,
            use_nwp=self.dataloader_config.nwp,
            use_sat=self.dataloader_config.sat,
            use_hrv=self.dataloader_config.hrv,
            use_pv=True,
            use_topo=self.dataloader_config.topo,
            size=self.dataloader_config.size,
            use_future=self.dataloader_config.use_future,
            batch_size=self.dataloader_config.batch,
        )
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(datapipe.collate().set_length(8000),
                          num_workers=self.dataloader_config.num_workers, batch_size=None)

