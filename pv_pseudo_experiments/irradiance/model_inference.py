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
        self.files = filter(lambda x: x, glob.glob(os.path.join(self.path_to_files,"*.npy")))
        self.files = list(self.files)
        f = filter(lambda x: x, glob.glob(os.path.join("/mnt/storage_u2_4tb_b/irradiance/*.npy")))
        self.files = self.files + list(f)
        self.files.sort()
        self.num_files = len(self.files)
        self.batch_size = batch_size
        self.transform = CenterCrop(16)

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
                    data = np.load(file, allow_pickle=True)
                    # split into x, y and meta
                    x = data[0][0]
                    y = data[2][0]
                    meta = data[1][0]
                    pv_meta = data[3]
                    location_data = data[4]
                    # Need to normalize metadata
                    meta[0] /= 90.
                    meta[1] /= 365.
                    # Divide by 10 for channels 22 and 23
                    x[22] *= 0
                    x[23] *= 0
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
                    pv_metas.append(pv_meta)
                    location_datas.append(location_data)
                x = torch.cat(xs, dim=0)
                y = torch.cat(ys, dim=0)
                meta = torch.cat(metas, dim=0)
                yield x, meta, y, pv_metas, location_datas
                files = []


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
        state_dict = torch.load("/home/jacob/best_irradiance_longest.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)

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

    def test_step(self, batch, batch_idx, **kwargs):
        x, meta, y, pv_metas, location_datas = batch
        x = x.half()
        meta = meta.half()
        y_hat = self(x, meta)
        # Outputs are the latents
        y_hat = y_hat.detach().cpu().numpy()
        # Save out the predictions to disk
        np.savez(f"/mnt/storage_ssd_4tb/irradiance_inference_outputs_new/{location_datas[0][0]}_{pv_metas[0][0][0]}.npz", latents=y_hat, pv_metas=pv_metas, location_datas=location_datas)
        return batch_idx

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        # Return your dataloader for training
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_ssd_4tb/irradiance_ones", train=True, batch_size=self.dataloader_config.batch)
        #rs = MultiProcessingReadingService(num_workers=self.dataloader_config.num_workers,
        #                                   multiprocessing_context="spawn")
        return DataLoader(dataset,num_workers=self.dataloader_config.num_workers, batch_size=None, collate_fn=collatey)

    def test_dataloader(self):
        dataset = PseudoIrradianceDataset(path_to_files="/mnt/storage_u2_4tb_b/irradiance", train=False, batch_size=1)
        return DataLoader(dataset,num_workers=0, batch_size=None, collate_fn=collatey)


