import datetime
import torch
import numpy as np
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe
from torch.utils.data.dataloader import DataLoader
import random
import os

testing = True


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def collaty(x):
    return x

class BatchWriter():
    def __init__(self, dataloader_config):
        self.dataloader_config = dataloader_config
        datapipe = pseudo_irradiance_datapipe(
            dataloader_config.config,
            start_time=datetime.datetime(2021, 1, 1) if testing else datetime.datetime(2018, 1, 1),
            end_time=datetime.datetime(2021, 12, 31) if testing else datetime.datetime(2020, 12, 31),
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
            one_d=True,
            is_test=testing,
        )
        self.dataloader = DataLoader(datapipe,
                  num_workers=self.dataloader_config.num_workers // 4 if testing else self.dataloader_config.num_workers, batch_size=None, collate_fn=collaty)

    def __call__(self):
        count = 0
        # Sometimes validation errors occur when picking sites to near the edge of the map, so reset it
        while True:
            try:
                for i, batch in enumerate(self.dataloader):
                    batch = batch[0], batch[1], batch[2], batch[3], batch[4]
                    while os.path.exists(f"/mnt/storage_u2_4tb_b/irradiance/{'test' if testing else 'train'}_{count}.npy"):
                        count += 1
                    np.save(f"/mnt/storage_u2_4tb_b/irradiance/{'test' if testing else 'train'}_{count}.npy", batch)
            except:
                datapipe = pseudo_irradiance_datapipe(
                    self.dataloader_config.config,
                    start_time=datetime.datetime(2021, 1, 1) if testing else datetime.datetime(2018, 1, 1),
                    end_time=datetime.datetime(2021, 12, 31) if testing else datetime.datetime(2020, 12, 31),
                    use_sun=self.dataloader_config.sun,
                    use_nwp=self.dataloader_config.nwp,
                    use_sat=self.dataloader_config.sat,
                    use_hrv=self.dataloader_config.hrv,
                    use_pv=True,
                    use_topo=self.dataloader_config.topo,
                    size=self.dataloader_config.size,
                    size_meters=self.dataloader_config.size_meters,
                    use_meters=self.dataloader_config.use_meters,
                    use_future=self.dataloader_config.use_future,
                    batch_size=self.dataloader_config.batch,
                    one_d=True,
                    is_test=testing,
                )
                self.dataloader = DataLoader(datapipe,
                                     num_workers=self.dataloader_config.num_workers // 4 if testing else self.dataloader_config.num_workers, batch_size=None, collate_fn=collaty)

