import torch.utils.data
import pandas as pd
import numpy as np
import os


class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str):
        self.sample_frame = pd.read_csv(root_path + "/data.csv")
        self.root_path = root_path


    def __len__(self):
        return len(self.sample_frame)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_path,
                                self.sample_frame["filename"][idx])
        image = np.load(img_name)
        poses = self.sample_frame.iloc[idx, 2:]
        poses = np.array([poses]).squeeze(axis=0)
        sample = {'image': image, 'poses': poses}

        return sample


if __name__ == "__main__":
    dataset = MemoryEfficientDataset("data/set_1")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['image'].shape, sample['poses'].shape)
