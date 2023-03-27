from ocf_datapipes.training.metnet_pv_site import metnet_site_datapipe

import matplotlib

from metnet.models import MetNetSingleShot

matplotlib.use("agg")
import warnings

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import datetime

import einops
import hydra
from omegaconf import DictConfig, OmegaConf


warnings.filterwarnings("ignore")

def mse_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MSE for each forecast horizon

    Args:
        output: The model estimate of size (batch_size, forecast_length)
        target: The truth of size (batch_size, forecast_length)

    Returns: A tensor of size (forecast_length)

    """
    return torch.mean((output - target) ** 2, dim=0)


def mae_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MAE for each forecast horizon

    Args:
        output: The model estimate of size (batch_size, forecast_length)
        target: The truth of size (batch_size, forecast_length)

    Returns: A tensor of size (forecast_length)

    """
    return torch.mean(torch.abs(output - target), dim=0)


torch.set_float32_matmul_precision("medium")

class LitMetNetModel(pl.LightningModule):
    def __init__(
        self,
        input_channels=42,
        center_crop_size=64,
        input_size=256,
        forecast_steps=70,
        hidden_dim=2048,
        att_layers=2,
        lr=1e-4,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.learning_rate = lr
        self.model = MetNetSingleShot(
            output_channels=forecast_steps,
            input_channels=input_channels,
            center_crop_size=center_crop_size,
            input_size=input_size,
            forecast_steps=forecast_steps,
            use_preprocessor=False,
            num_att_layers=att_layers,
            hidden_dim=hidden_dim,
        )
        self.pooler = torch.nn.AdaptiveAvgPool2d(1)
        self.config = self.model.config
        self.save_hyperparameters()

    def forward(self, x):
        return F.relu(self.pooler(self.model(x))[:, :, 0, 0])

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, y = batch
        y = y[0]
        x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
        y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
        x = x.half()
        y = y.half()
        y = y[:, 1:, 0]  # Take out the T0 output
        y_hat = self(x)
        # loss = self.weighted_losses.get_mse_exp(y_hat, y)
        # self.log("loss", loss)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        self.log("loss", loss)
        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
            # on_step=True,
            # on_epoch=True,
            # sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        # add metrics for each forecast horizon
        mse_each_forecast_horizon_metric = mse_each_forecast_horizon(output=y_hat, target=y)
        mae_each_forecast_horizon_metric = mae_each_forecast_horizon(output=y_hat, target=y)

        metrics_mse = {
            f"MSE_forecast_horizon_{i}/{tag}": mse_each_forecast_horizon_metric[i]
            for i in range(self.forecast_steps)
        }
        metrics_mae = {
            f"MAE_forecast_horizon_{i}/{tag}": mae_each_forecast_horizon_metric[i]
            for i in range(self.forecast_steps)
        }

        self.log_dict(
            {**metrics_mse, **metrics_mae},
            # on_step=True,
            # on_epoch=True,
            # sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if batch_idx % 100 == 0:  # Log every 100 batches
            self.log_tb_images((x, y, y_hat, [batch_idx for _ in range(x.shape[0])]))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def log_tb_images(self, viz_batch) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

            # Log the images (Give them different names)
        for img_idx, (_, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            fig = plt.figure()
            plt.plot(list(range(360)), y_pred.cpu().detach().numpy(), label="Forecast")
            plt.plot(list(range(360)), y_true.cpu().detach().numpy(), label="Truth")
            plt.title("GT vs Pred PV Site Single Shot")
            plt.legend(loc="best")
            tb_logger.add_figure(f"GT_Vs_Pred/{img_idx}", fig, batch_idx)


def create_metnet_train_dataloader(config: DictConfig):
    return metnet_site_datapipe(
        config.config,
        start_time=datetime.datetime(2014, 1, 1),
        end_time=datetime.datetime(2020, 12, 31),
        use_sun=config.sun,
        use_nwp=config.nwp,
        use_sat=config.sat,
        use_hrv=config.hrv,
        use_pv=True,
        use_topo=config.topo,
        pv_in_image=True,
        output_size=config.size,
        center_size_meters=config.center_meter,
        context_size_meters=config.context_meter,
    )

def create_metnet_test_dataloader(config: DictConfig):
    return metnet_site_datapipe(
        config.config,
        start_time=datetime.datetime(2021, 1, 1),
        end_time=datetime.datetime(2021, 12, 31),
        use_sun=config.sun,
        use_nwp=config.nwp,
        use_sat=config.sat,
        use_hrv=config.hrv,
        use_pv=True,
        use_topo=config.topo,
        pv_in_image=True,
        output_size=config.size,
        center_size_meters=config.center_meter,
        context_size_meters=config.context_meter,
    )
