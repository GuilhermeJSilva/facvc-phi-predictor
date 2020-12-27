import torch.utils.data
import pandas as pd
import numpy as np
import os


class PhaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, csv_path: str):
        self.sample_frame = pd.read_csv(csv_path)
        self.root_path = root_path

    def __len__(self):
        return len(self.sample_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_path, self.sample_frame["filename"][idx])
        image = np.expand_dims(np.load(img_name), 0)
        image = np.log(image, where=image != 0)
        image = torch.from_numpy(image).double()
        poses = self.sample_frame.iloc[idx, 2:]
        poses = torch.tensor([poses]).squeeze(axis=0).double()

        return image, poses


if __name__ == "__main__":
    dataset = PhaseDataset("data/set_2/samples", "data/set_2/data.csv")
    for i in range(len(dataset)):
        image, poses = dataset[i]
        print(i, image.shape, poses.shape)
