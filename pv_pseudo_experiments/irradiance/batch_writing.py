import datetime
import torch
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from torch.utils.data.dataloader import DataLoader
import os

class BatchWriter():
    def __init__(self, dataloader_config):
        datapipe = pseudo_irradiance_datapipe(
            dataloader_config.config,
            start_time=datetime.datetime(2008, 1, 1),
            end_time=datetime.datetime(2021, 12, 31),
            use_sun=dataloader_config.sun,
            use_nwp=dataloader_config.nwp,
            use_sat=dataloader_config.sat,
            use_hrv=dataloader_config.hrv,
            use_pv=True,
            use_topo=dataloader_config.topo,
            size=dataloader_config.size,
            size_meters=dataloader_config.size_meters,
            use_meters=dataloader_config.use_meters,
            use_future=dataloader_config.use_future,
            batch_size=dataloader_config.batch,
        )
        self.dataloader = DataLoader(datapipe.set_length(15000),
                  num_workers=dataloader_config.num_workers, batch_size=None)

    def __call__(self):
        count = 0
        for i, batch in enumerate(self.dataloader):
            batch = (torch.stack(batch[0], dim=0), torch.stack(batch[1], dim=0), torch.stack(batch[2], dim=0))
            while os.path.exists(f"/mnt/storage_ssd_4tb/irradiance_batches/sun/2021_{count}.pth"):
                count += 1
            torch.save(batch, f"/mnt/storage_ssd_4tb/irradiance_batches/sun/2021_{count}.pth")
