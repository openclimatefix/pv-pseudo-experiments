import einops
import torch
from torch.utils.data.dataset import IterableDataset, Dataset
import glob
import os
from torch.utils.data.dataloader import DataLoader


class PseudoIrradianceDataset(IterableDataset):
    # take as an init the folder containing .pth files and then load them in the __iter__ method and split into train and val
    def __init__(self, path_to_files: str, train: bool = True):
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

    def __len__(self):
        return self.num_files

    def __iter__(self):
        for file in self.files:
            # load file using torch.load
            data = torch.load(file)
            # split into x, y and meta
            x = data[0]
            y = data[2]
            meta = data[1]
            # yield x, y and meta
            # Use einops to split the first dimension into batch size of 4 and then channels
            x = einops.rearrange(x, "(b t) c h w -> b c t h w", b=4)
            y = einops.rearrange(y, "(b t c) h w -> b c t h w", b=4, c=1)
            meta = einops.rearrange(meta, "(b c) h w -> b c h w", b=4)
            #x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
            #y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
            #meta = torch.nan_to_num(input=meta, posinf=1.0, neginf=0.0)
            mask = meta > 0.0
            if torch.all(mask == False):
                continue
            yield x, meta, y

dataset = PseudoIrradianceDataset(path_to_files="/run/media/jacob/data/irradiance_batches/irradiance_batches/", train=True)
dataloader = DataLoader(dataset,num_workers=0, batch_size=None)
import matplotlib.pyplot as plt

for batch in iter(dataloader):
    x, meta, y = batch
    # Plot all of x channels in the same figure that has 5 rows
    for time in range(x.shape[2]):
        fig, axs = plt.subplots(6, 6, figsize=(20, 20))
        for channel in range(x.shape[1]):
            row_index = channel // 6
            column_index = channel % 6
            axs[row_index, column_index].imshow(x[0,channel,time,:,:])
            axs[row_index, column_index].set_title("X Channel: " + str(channel))
        fig.suptitle("Time: " + str(time))
        fig.show()
    # Now plot the same but with the difference between each time step
    for time in range(x.shape[2]-1):
        fig, axs = plt.subplots(6, 6, figsize=(20, 20))
        for channel in range(x.shape[1]):
            row_index = channel // 6
            column_index = channel % 6
            axs[row_index, column_index].imshow(x[0,channel,time+1,:,:] - x[0,channel,time,:,:])
            axs[row_index, column_index].set_title("X Channel: " + str(channel))
        fig.suptitle("Difference: " + str(time))
        fig.show()
    exit()
    plt.imshow(meta[0,0,:,:])
    plt.show()
    plt.imshow(y[0,0,0,:,:])
    plt.show()
    break
