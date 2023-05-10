import datetime
from ocf_datapipes.training.metnet_pv_site import metnet_site_datapipe
import torch
from torch.utils.data.dataloader import DataLoader
import os

class PVBatchWriter():
    def __init__(self, dataloader_config):
        datapipe = metnet_site_datapipe(
            dataloader_config.config,
            start_time=datetime.datetime(2014, 1, 1),
            end_time=datetime.datetime(2020, 12, 31),
            use_sun=dataloader_config.sun,
            use_nwp=dataloader_config.nwp,
            use_sat=dataloader_config.sat,
            use_hrv=dataloader_config.hrv,
            use_pv=True,
            use_topo=dataloader_config.topo,
            pv_in_image=True,
            output_size=dataloader_config.size,
            center_size_meters=dataloader_config.center_meter,
            context_size_meters=dataloader_config.context_meter,
            batch_size=dataloader_config.batch,
            match_simon=dataloader_config.match_simon,)
        self.dataloader = DataLoader(datapipe,
                  num_workers=dataloader_config.num_workers, batch_size=None)

    def __call__(self):
        count = 0
        for i, batch in enumerate(self.dataloader):
            #batch = (torch.stack(batch[0], dim=0), torch.stack(batch[1], dim=0))
            while os.path.exists(f"/mnt/storage_ssd_4tb/metnet_batches/train_{count}.pth"):
                count += 1
            torch.save(batch, f"/mnt/storage_ssd_4tb/metnet_batches/train_{count}.pth")
