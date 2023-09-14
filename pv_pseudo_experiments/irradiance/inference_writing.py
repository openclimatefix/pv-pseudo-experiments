import datetime
import torch
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np

def collaty(x):
    return x

class BatchWriter():
    def __init__(self, dataloader_config):
        datapipe = pseudo_irradiance_datapipe(
            dataloader_config.config,
            start_time=datetime.datetime(2020, 1, 1),
            end_time=datetime.datetime(2020, 12, 31),
            use_sun=dataloader_config.sun,
            use_nwp=dataloader_config.nwp,
            use_sat=dataloader_config.sat,
            use_hrv=dataloader_config.hrv,
            use_pv=True,
            use_topo=dataloader_config.topo,
            size=dataloader_config.size,
            size_meters=dataloader_config.size_meters,
            use_meters=False,
            use_future=dataloader_config.use_future,
            batch_size=dataloader_config.batch,
            one_d=True,
            is_test=True,
        )
        self.dataloader = DataLoader(datapipe,
                  num_workers=dataloader_config.num_workers, batch_size=None, collate_fn=collaty)

    def __call__(self):
        count = 0
        for i, batch in enumerate(self.dataloader):
            batch = batch[0], batch[1], batch[2], batch[3], batch[4]
            if os.path.exists(f"/mnt/storage_ssd_4tb/irradiance_inference_2020/test_{count}.npy"):
                count += 1
                continue
            np.save(f"/mnt/storage_ssd_4tb/irradiance_inference_2020/test_{count}.npy", batch)
