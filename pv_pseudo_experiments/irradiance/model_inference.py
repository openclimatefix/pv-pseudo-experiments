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

def collatey(x):
    return x

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
        files = []
        for f in self.files:
            # load file using torch.load
            files.append(f)
            if len(files) == self.batch_size:
                xs = []
                ys = []
                metas = []
                pv_metas = []
                location_datas = []
                for file in files:
                    data = torch.load(file)
                    # split into x, y and meta
                    x = data[0]
                    y = data[2]
                    meta = data[1]
                    pv_meta = data[3]
                    location_data = data[4]
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
                    pv_metas.append(pv_meta)
                    location_datas.append(location_data)
                x = torch.cat(xs, dim=0)
                y = torch.cat(ys, dim=0)
                meta = torch.cat(metas, dim=0)
                yield x, meta, y, pv_metas, location_datas


class LitIrradianceInferenceModel(LightningModule):
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
        return self.model(x, meta, output_latents=True)

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, meta, y, pv_metas, location_datas = batch
        y_hat = self(x, meta)
        # Add in single channel output
        y_hat = einops.repeat(y_hat, "b t -> b t c", c=1)

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
        return loss

    def test_step(self, batch, **kwargs):
        x, meta, y, pv_metas, location_datas = batch
        y_hat = self(x, meta)
        # Outputs are the latents
        y_hat = y_hat.detach().cpu().numpy()
        # Save out the predictions to disk
        np.savez(f"/mnt/storage_ssd_4tb/irradiance_inference_outputs_2020/{location_datas[0]}_{pv_metas[0][0]}.npz", latents=y_hat, pv_metas=pv_metas, location_datas=location_datas)
        return 0.0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        # Return your dataloader for training
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_ssd_4tb/irradiance_ones", train=True, batch_size=self.dataloader_config.batch)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(dataset,num_workers=self.dataloader_config.num_workers, batch_size=None, collate_fn=collatey)

    def test_dataloader(self):
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_ssd_4tb/irradiance_inference_2020", train=False, batch_size=self.dataloader_config.batch)
        return DataLoader(dataset,num_workers=self.dataloader_config.num_workers, batch_size=None, collate_fn=collatey)

