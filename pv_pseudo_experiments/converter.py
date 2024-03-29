import datetime

import einops
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig
from pseudo_labeller.model import PsuedoIrradienceForecastor
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
import glob
import os
from random import shuffle
import numpy as np

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
            input_channels=30,
            input_size=16,
            input_steps=14,
            output_channels=16,
            conv3d_channels=256,
            hidden_dim=16,
            kernel_size=3,
            num_layers=4,
            output_steps=48,
            pv_meta_input_channels=2,
        )
        self.config = self.model.config
        self.save_hyperparameters()

    def forward(self, x, meta):
        return self.model(x, meta, output_latents=False)

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, meta, y = batch
        # Convert to same dtype as model
        x = x.to(dtype=torch.float16)
        y = y.to(dtype=torch.float16)
        meta = meta.to(dtype=torch.float16)
        y_hat = self(x, meta)
        # Add in single channel output
        y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

        #mask = meta > 0.0
        #mask = torch.unsqueeze(torch.sum(mask, dim=1) > 0.0, dim=1)

        # Expand to match the ground truth shape
        #mask = einops.repeat(mask, "b c h w -> b c t h w", t=y.shape[2])

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("train/loss", loss)
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
        )
        #if batch_idx % 100 == 0:  # Log every 100 batches
        #    self.log_tb_images((x[:20], y[:20], y_hat[:20], meta[:20], [batch_idx for _ in range(min(x.shape[0],20))]), tag=tag)
        return loss

    def validation_step(self, batch, batch_idx):
        tag = "val"
        x, meta, y = batch
        # Convert to same dtype as model
        x = x.to(dtype=torch.float16)
        y = y.to(dtype=torch.float16)
        meta = meta.to(dtype=torch.float16)
        y_hat = self(x, meta)
        # Add in single channel output
        y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

        #mask = meta > 0.0
        #mask = torch.unsqueeze(torch.sum(mask, dim=1) > 0.0, dim=1)

        # Expand to match the ground truth shape
        #mask = einops.repeat(mask, "b c h w -> b c t h w", t=y.shape[2])

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("val/loss", loss)
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
        )
        #if batch_idx % 100 == 0:  # Log every 100 batches
        #    self.log_tb_images((x[:20], y[:20], y_hat[:20], meta[:20], [batch_idx for _ in range(min(x.shape[0],20))]), tag=tag)
        return loss

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
        for img_idx, (x, y_true, y_pred, meta, batch_idx) in enumerate(zip(*viz_batch)):
            # Plot out the x and y image sequences to matplotlib figure
            # Pick random example from the batch to do the plotting
            # rand_idx = np.random.randint(0, x.shape[0])

            fig = plt.figure(figsize=(10, 20))
            axs = fig.subplots(x.shape[0], x.shape[1])
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    axs[i, j].imshow(x[i, j, :, :].cpu().detach().numpy())
                    axs[i, j].set_title(f"T: {j} C: {i}")
                    axs[i, j].axis("off")
            tb_logger.add_figure(f"Input/{tag}/{img_idx}", fig, batch_idx)
            wandb_logger.log({f"Input/{tag}/{img_idx}": fig})
            fig.close()
            # Forecast steps
            fig = plt.figure(figsize=(10, 30))
            axs = fig.subplots(1, y_true.shape[1])
            for j in range(y_true.shape[1]):
                axs[0, j].imshow(y_true[0, j,].cpu().detach().numpy())
                axs[0, j].axis("off")
            tb_logger.add_figure(f"GT/{tag}/{img_idx}", fig, batch_idx)
            wandb_logger.log({f"GT/{tag}/{img_idx}": fig})
            fig.close()
            # Forecast steps predicted
            #fig = plt.figure(figsize=(10, 30))
            #axs = fig.subplots(1, y_pred.shape[1])
            #for j in range(y_pred.shape[1]):
            #    axs[0, j].imshow(y_pred[0, j, :, :].cpu().detach().numpy())
            #    axs[0, j].axis("off")
            #tb_logger.add_figure(f"Pred/{tag}/{img_idx}", fig, batch_idx)
            #wandb_logger.log({f"Pred/{tag}/{img_idx}": fig})
            #fig.close()

            # Now use meta to plot out the GT vs Pred for the pixels that have values
            #mask = meta > 0.0
            #mask = torch.unsqueeze(torch.sum(mask, dim=0) > 0.0, dim=1)

            # Randomly take 10 pixels indicies where the mask is true
            #mask = mask.squeeze()
            #mask = mask.nonzero(as_tuple=True)
            #permutation_idx = torch.randperm(mask[0].shape[0])[:10]
            #x_mask = mask[0][permutation_idx]
            #y_mask = mask[1][permutation_idx]
            # Select those 10 pixels in y_true and y_pred and plot them
            # Mask is 2 tensors now I think
            for example_idx in range(min(y_true.shape[0], 20)):
                gt = y_true[example_idx].cpu().detach().numpy()
                pred = y_pred[example_idx].cpu().detach().numpy()
                # Now this is 2 1D arrays, plot them on matplotlib figure
                fig = plt.figure()
                plt.plot(list(range(gt.shape[0])), gt, label="GT")
                plt.plot(list(range(gt.shape[0])), pred, label="Pred")
                plt.title(f"GT vs Pred for Pixel")
                plt.legend(loc="best")
                tb_logger.add_figure(f"GT_Vs_Pred/{tag}/{img_idx}_{example_idx}", fig, batch_idx)
                wandb_logger.log({f"GT_Vs_Pred/{tag}/{img_idx}_{example_idx}": fig})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

full_model = LitIrradianceModel.load_from_checkpoint("/home/jacob/irradiance_best_longest.ckpt")
model = full_model.model

torch.save(model.state_dict(), "/home/jacob/best_irradiance_longest.pt")
import datetime
exit()
import einops
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from omegaconf import DictConfig
from pseudo_labeller.model import PsuedoIrradienceForecastor
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
import glob
import os
from random import shuffle
import numpy as np

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
            input_channels=30,
            input_size=16,
            input_steps=14,
            output_channels=16,
            conv3d_channels=256,
            hidden_dim=16,
            kernel_size=3,
            num_layers=4,
            output_steps=48,
            pv_meta_input_channels=2,
        )
        self.config = self.model.config
        self.save_hyperparameters()

    def forward(self, x, meta):
        return self.model(x, meta, output_latents=False)

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, meta, y = batch
        # Convert to same dtype as model
        x = x.to(dtype=torch.float16)
        y = y.to(dtype=torch.float16)
        meta = meta.to(dtype=torch.float16)
        y_hat = self(x, meta)
        # Add in single channel output
        y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

        #mask = meta > 0.0
        #mask = torch.unsqueeze(torch.sum(mask, dim=1) > 0.0, dim=1)

        # Expand to match the ground truth shape
        #mask = einops.repeat(mask, "b c h w -> b c t h w", t=y.shape[2])

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("train/loss", loss)
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
        )
        #if batch_idx % 100 == 0:  # Log every 100 batches
        #    self.log_tb_images((x[:20], y[:20], y_hat[:20], meta[:20], [batch_idx for _ in range(min(x.shape[0],20))]), tag=tag)
        return loss

    def validation_step(self, batch, batch_idx):
        tag = "val"
        x, meta, y = batch
        # Convert to same dtype as model
        x = x.to(dtype=torch.float16)
        y = y.to(dtype=torch.float16)
        meta = meta.to(dtype=torch.float16)
        y_hat = self(x, meta)
        # Add in single channel output
        y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

        #mask = meta > 0.0
        #mask = torch.unsqueeze(torch.sum(mask, dim=1) > 0.0, dim=1)

        # Expand to match the ground truth shape
        #mask = einops.repeat(mask, "b c h w -> b c t h w", t=y.shape[2])

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        if torch.isinf(loss) or torch.isnan(loss):
            raise ValueError("Loss is NaN or Inf. Exiting.")
        self.log("val/loss", loss)
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
        )
        #if batch_idx % 100 == 0:  # Log every 100 batches
        #    self.log_tb_images((x[:20], y[:20], y_hat[:20], meta[:20], [batch_idx for _ in range(min(x.shape[0],20))]), tag=tag)
        return loss

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
        for img_idx, (x, y_true, y_pred, meta, batch_idx) in enumerate(zip(*viz_batch)):
            # Plot out the x and y image sequences to matplotlib figure
            # Pick random example from the batch to do the plotting
            # rand_idx = np.random.randint(0, x.shape[0])

            fig = plt.figure(figsize=(10, 20))
            axs = fig.subplots(x.shape[0], x.shape[1])
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    axs[i, j].imshow(x[i, j, :, :].cpu().detach().numpy())
                    axs[i, j].set_title(f"T: {j} C: {i}")
                    axs[i, j].axis("off")
            tb_logger.add_figure(f"Input/{tag}/{img_idx}", fig, batch_idx)
            wandb_logger.log({f"Input/{tag}/{img_idx}": fig})
            fig.close()
            # Forecast steps
            fig = plt.figure(figsize=(10, 30))
            axs = fig.subplots(1, y_true.shape[1])
            for j in range(y_true.shape[1]):
                axs[0, j].imshow(y_true[0, j,].cpu().detach().numpy())
                axs[0, j].axis("off")
            tb_logger.add_figure(f"GT/{tag}/{img_idx}", fig, batch_idx)
            wandb_logger.log({f"GT/{tag}/{img_idx}": fig})
            fig.close()
            # Forecast steps predicted
            #fig = plt.figure(figsize=(10, 30))
            #axs = fig.subplots(1, y_pred.shape[1])
            #for j in range(y_pred.shape[1]):
            #    axs[0, j].imshow(y_pred[0, j, :, :].cpu().detach().numpy())
            #    axs[0, j].axis("off")
            #tb_logger.add_figure(f"Pred/{tag}/{img_idx}", fig, batch_idx)
            #wandb_logger.log({f"Pred/{tag}/{img_idx}": fig})
            #fig.close()

            # Now use meta to plot out the GT vs Pred for the pixels that have values
            #mask = meta > 0.0
            #mask = torch.unsqueeze(torch.sum(mask, dim=0) > 0.0, dim=1)

            # Randomly take 10 pixels indicies where the mask is true
            #mask = mask.squeeze()
            #mask = mask.nonzero(as_tuple=True)
            #permutation_idx = torch.randperm(mask[0].shape[0])[:10]
            #x_mask = mask[0][permutation_idx]
            #y_mask = mask[1][permutation_idx]
            # Select those 10 pixels in y_true and y_pred and plot them
            # Mask is 2 tensors now I think
            for example_idx in range(min(y_true.shape[0], 20)):
                gt = y_true[example_idx].cpu().detach().numpy()
                pred = y_pred[example_idx].cpu().detach().numpy()
                # Now this is 2 1D arrays, plot them on matplotlib figure
                fig = plt.figure()
                plt.plot(list(range(gt.shape[0])), gt, label="GT")
                plt.plot(list(range(gt.shape[0])), pred, label="Pred")
                plt.title(f"GT vs Pred for Pixel")
                plt.legend(loc="best")
                tb_logger.add_figure(f"GT_Vs_Pred/{tag}/{img_idx}_{example_idx}", fig, batch_idx)
                wandb_logger.log({f"GT_Vs_Pred/{tag}/{img_idx}_{example_idx}": fig})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

full_model = LitIrradianceModel.load_from_checkpoint("/home/jacob/best_irradiance.ckpt")
model = full_model.model

torch.save(model.state_dict(), "/home/jacob/best_irradiance.pt")

