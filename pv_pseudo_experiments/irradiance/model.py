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
from torchvision.transforms import CenterCrop

class PseudoIrradianceDataset(IterableDataset):
    # take as an init the folder containing .pth files and then load them in the __iter__ method and split into train and val
    def __init__(self, path_to_files: str, train: bool = True, batch_size: int = 4, crop_size=128):
        super().__init__()
        self.path_to_files = path_to_files
        self.train = train
        # Use glob to get all files in the path_to_files and filter out ones that have 2021 in them if train is true
        # and ones that have 2020 in them if train is false
        # use the filter function and the lambda function to do this
        if self.train:
            self.files = filter(lambda x: "test" not in x, glob.glob(os.path.join(self.path_to_files,"*.npy")))
            files2 = filter(lambda x: "test" not in x, glob.glob(os.path.join("/mnt/storage_u2_4tb_b/irradiance","*.npy")))
        else:
            self.files = filter(lambda x: "test" in x, glob.glob(os.path.join(self.path_to_files,"*.npy")))
            files2 = []
        self.files = list(self.files) + list(files2)
        self.files.sort()
        self.num_files = len(self.files)
        self.batch_size = batch_size
        self.transform = CenterCrop(crop_size)

    def __len__(self):
        return self.num_files // self.batch_size

    def __iter__(self):
        while True:
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
                        data = np.load(file, allow_pickle=True)
                        # split into x, y and meta
                        x = data[0][0]
                        y = data[2][0]
                        meta = data[1][0]
                        # Need to normalize metadata
                        meta[0] /= 90.
                        meta[1] /= 365.                        
                        # Divide by 10 for channels 22 and 23
                        x[22] *= 0
                        x[23] *= 0
                        # yield x, y and meta
                        # Use einops to split the first dimension into batch size of 4 and then channels
                        #y = einops.rearrange(y, "(b c) h w -> b c h w", c=1)
                        x = torch.from_numpy(x)
                        y = torch.from_numpy(y)
                        meta = torch.from_numpy(meta)
                        x = torch.unsqueeze(torch.nan_to_num(input=x, posinf=1.0, neginf=0.0), dim=0)
                        y = torch.unsqueeze(torch.nan_to_num(input=y, posinf=1.0, neginf=0.0), dim=0)
                        if y.shape[1] % 3 != 0:
                            y = y[:, :-(y.shape[1] % 3)] # Make it divisible by 3
                        y = torch.mean(y.reshape(-1, 3), dim=1) # Average over 3 timesteps
                        #y = y[:,:] # Only keep first 2 hours
                        meta = torch.nan_to_num(input=meta, posinf=1.0, neginf=0.0)
                        meta = torch.squeeze(meta)
                        meta = torch.unsqueeze(meta, dim=0)
                        x = self.transform(x)
                        xs.append(x)
                        ys.append(y)
                        metas.append(meta)
                    x = torch.cat(xs, dim=0)
                    y = torch.stack(ys, dim=0)
                    meta = torch.cat(metas, dim=0)
                    yield x, meta, y
                    files = []


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
        # Convert to same dtype as model
        x = x.to(dtype=torch.float16)
        y = y.to(dtype=torch.float16)
        meta = meta.to(dtype=torch.float16)
        y_hat = self(x, meta)
        # Add in single channel output
        #y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

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
        #print(f"{x.shape=},{y.shape=},{meta.shape=}")
        #exit()
        y_hat = self(x, meta)
        # Add in single channel output
        #y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

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
        if batch_idx % 100 == 0:  # Log every 100 batches
            #print(f"{y.shape=},{y_hat.shape=}")
            #exit()
            self.log_tb_images((x[:20], y[:20], y_hat[:20], meta[:20], [batch_idx for _ in range(min(x.shape[0],20))]), tag=tag)
        return loss

    def log_tb_images(self, viz_batch, tag) -> None:

        # Get tensorboard logger
        #tb_logger = None
        #for logger in self.trainer.loggers:
        #    if isinstance(logger, TensorBoardLogger):
        #        tb_logger = logger.experiment
        #        break
        wandb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger.experiment
                break

        #if tb_logger is None:
        #    raise ValueError("TensorBoard Logger not found")

            # Log the images (Give them different names)
        for img_idx, (x, y_true, y_pred, meta, batch_idx) in enumerate(zip(*viz_batch)):
            # Plot out the x and y image sequences to matplotlib figure
            # Pick random example from the batch to do the plotting
            # rand_idx = np.random.randint(0, x.shape[0])

            #fig = plt.figure(figsize=(10, 20))
            #axs = fig.subplots(x.shape[0], x.shape[1])
            #for i in range(x.shape[0]):
            #    for j in range(x.shape[1]):
            #        axs[i, j].imshow(x[i, j, :, :].cpu().detach().numpy())
            #        axs[i, j].set_title(f"T: {j} C: {i}")
            #        axs[i, j].axis("off")
            #tb_logger.add_figure(f"Input/{tag}/{img_idx}", fig, batch_idx)
            #wandb_logger.log({f"Input/{tag}/{img_idx}": fig})
            #fig.close()
            # Forecast steps
            #fig = plt.figure(figsize=(10, 30))
            #axs = fig.subplots(1, y_true.shape[1])
            #for j in range(y_true.shape[1]):
            #    axs[0, j].imshow(y_true[0, j,].cpu().detach().numpy())
            #    axs[0, j].axis("off")
            #tb_logger.add_figure(f"GT/{tag}/{img_idx}", fig, batch_idx)
            #wandb_logger.log({f"GT/{tag}/{img_idx}": fig})
            #fig.close()
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
            #for example_idx in range(8):
            gt = y_true.cpu().detach().numpy()
            pred = y_pred.cpu().detach().numpy()
            #print(f"{y_true.shape=},{gt.shape=},{y_pred.shape=},{pred.shape=}")
            #exit()
            # Now this is 2 1D arrays, plot them on matplotlib figure
            fig = plt.figure()
            plt.plot(list(range(gt.shape[0])), gt, label="GT")
            plt.plot(list(range(gt.shape[0])), pred, label="Pred")
            plt.title(f"GT vs Pred")
            plt.legend(loc="best")
            #tb_logger.add_figure(f"GT_Vs_Pred/{tag}/{img_idx}_{example_idx}", fig, batch_idx)
            wandb_logger.log({f"GT_Vs_Pred/{tag}/{img_idx}": fig})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        # Return your dataloader for training
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_u2_4tb_a/irradiance/", train=True, batch_size=self.dataloader_config.batch, crop_size=self.dataloader_config.size)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(dataset,num_workers=self.dataloader_config.num_workers, batch_size=None)
    def val_dataloader(self):
        # Return your dataloader for training
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_u2_4tb_a/irradiance/", train=False, batch_size=self.dataloader_config.batch, crop_size=self.dataloader_config.size)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(dataset,num_workers=self.dataloader_config.num_workers, batch_size=None)


