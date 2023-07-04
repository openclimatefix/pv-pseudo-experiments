import matplotlib
from metnet.models import MetNetSingleShot
#from ocf_datapipes.training.metnet_pv_site import metnet_site_datapipe
matplotlib.use("agg")
import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torch.utils.data.dataloader import DataLoader
import glob
from random import shuffle
from torch.utils.data import IterableDataset, default_collate
import os
import einops

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
class PseudoIrradianceDataset(IterableDataset):
    # take as an init the folder containing .pth files and then load them in the __iter__ method and split into train and val
    def __init__(self, path_to_files: str, train: bool = True, batch_size: int = 4):
        super().__init__()
        self.path_to_files = path_to_files
        self.train = train
        # Use glob to get all files in the path_to_files and filter out ones that have 2021 in them if train is true
        # and ones that have 2020 in them if train is false
        # use the filter function and the lambda function to do this
        if self.train:
            self.files = filter(lambda x: "2021" not in x, glob.glob(os.path.join(self.path_to_files,"*.pth")))
        else:
            self.files = filter(lambda x: "2021" in x, glob.glob(os.path.join(self.path_to_files,"*.pth")))
        self.files = list(self.files)
        self.files.sort()
        self.num_files = len(self.files)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_files // self.batch_size

    def __iter__(self):
        if self.train:
            self.files = filter(lambda x: "2021" not in x, glob.glob(os.path.join(self.path_to_files,"*.pth")))
            self.files = list(self.files)
            shuffle(self.files)
        files = []
        for f in self.files:
            # load file using torch.load
            files.append(f)
            if len(files) == self.batch_size:
                xs = []
                ys = []
                metas = []
                for file in files:
                    data = torch.load(file)
                    # split into x, y and meta
                    x = data[0]
                    y = data[2]
                    meta = data[1]
                    print(y.shape)
                    print(meta.shape)
                    print(x.shape)
                    # yield x, y and meta
                    # Use einops to split the first dimension into batch size of 4 and then channels
                    #y = einops.rearrange(y, "(b c) h w -> b c h w", c=1)
                    x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
                    y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
                    #if y.shape[1] % 3 != 0:
                    #    y = y[:, :-(y.shape[1] % 3)] # Make it divisible by 3
                    #y = torch.mean(y.reshape(-1, 3), dim=1) # Average over 3 timesteps
                    meta = torch.nan_to_num(input=meta, posinf=1.0, neginf=0.0)
                    xs.append(x)
                    ys.append(y)
                    metas.append(meta)
                x = torch.cat(xs, dim=0)
                y = torch.cat(ys, dim=0)
                meta = torch.cat(metas, dim=0)
                if self.train:
                    x = x[:,1:] # Remove PV history
                yield x, meta, y
                files = []


class LitMetNetModel(LightningModule):
    def __init__(
            self,
            config: DictConfig,
            dataloader_config: DictConfig,
    ):
        super().__init__()
        self.forecast_steps = config.forecast_steps
        self.learning_rate = config.lr
        if config.weights:
            self.model = MetNetSingleShot.from_pretrained(config.weights)
        else:
            self.model = MetNetSingleShot(
                output_channels=config.forecast_steps*config.irradiance_channels,
                input_channels=config.input_channels,
                center_crop_size=config.center_crop_size,
                input_size=config.input_size,
                forecast_steps=config.forecast_steps,
                use_preprocessor=False,
                num_att_layers=config.att_layers,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        self.pooler = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(config.irradiance_channels+2, config.hidden_dim) # Each one will have output of pseudo-irradiance channels + 2 metadata channels
        self.linear2 = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.linear3 = torch.nn.Conv1d(config.hidden_dim, config.forecast_steps, kernel_size=1)

        self.config = self.model.config
        self.dataloader_config = dataloader_config
        self.model_config = config
        self.save_hyperparameters()

    def forward(self, x, meta):
        output = self.pooler(self.model(x))
        # Combine with metadata to get output
        # Make meta same size as output
        # Want the inputs to be
        output = torch.cat([output, meta], dim=1)

        return F.relu(output)

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, y = batch
        #x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
        #y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
        x = x.half()
        y = y.half()
        #y = y[:, 1:, 0]  # Take out the T0 output
        y_hat = self(x)
        # loss = self.weighted_losses.get_mse_exp(y_hat, y)
        # self.log("loss", loss)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("train/loss", loss)
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
            self.log_tb_images((x, y, y_hat, [batch_idx for _ in range(x.shape[0])]), tag)
        return loss

    def validation_step(self, batch, batch_idx):
        tag = "val"
        x, y = batch
        #x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
        #y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
        x = x.half()
        y = y.half()
        #y = y[:, 1:, 0]  # Take out the T0 output
        y_hat = self(x)
        # loss = self.weighted_losses.get_mse_exp(y_hat, y)
        # self.log("loss", loss)
        #mask = y >= 0.05 # Should cancel out night time
        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("val/loss", loss)
        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
            on_step=True,
            on_epoch=True,
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
            on_step=True,
            on_epoch=True,
            # sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if batch_idx % 100 == 0:  # Log every 100 batches
            self.log_tb_images((x, y, y_hat, [batch_idx for _ in range(x.shape[0])]), tag)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def log_tb_images(self, viz_batch, tag) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        wandb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

            # Log the images (Give them different names)
        for img_idx, (_, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            fig = plt.figure()
            plt.plot(list(range(y_pred.shape[0])), y_pred.cpu().detach().numpy(), label="Forecast")
            plt.plot(list(range(y_true.shape[0])), y_true.cpu().detach().numpy(), label="Truth")
            plt.title("GT vs Pred PV Site Single Shot")
            plt.legend(loc="best")
            tb_logger.add_figure(f"GT_Vs_Pred/{tag}/{img_idx}", fig, batch_idx)
            if wandb_logger is not None:
                wandb_logger.log({f"GT_Vs_Pred/{tag}/{img_idx}": fig})

    def train_dataloader(self):
        # Return your dataloader for training
        datapipe = MetNetDataset(path_to_files="/mnt/storage_ssd_4tb/metnet_batches/", train=True, batch_size=self.dataloader_config.batch)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers, multiprocessing_context="spawn")
        return DataLoader(datapipe, num_workers=self.dataloader_config.num_workers, batch_size=None)

    def val_dataloader(self):
        # Return your dataloader for training
        datapipe = MetNetDataset(path_to_files="/mnt/storage_ssd_4tb/metnet_batches/", train=False, batch_size=self.dataloader_config.batch)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers, multiprocessing_context="spawn")
        return DataLoader(datapipe, num_workers=self.dataloader_config.num_workers, batch_size=None)

def convert_to_tensor(batch):
    # Each batch has 0 being the inputs, and 1 being the targets
    return (torch.tensor(batch[0], dtype=torch.float32),
            torch.squeeze(torch.tensor(batch[1], dtype=torch.float32), dim=1))
