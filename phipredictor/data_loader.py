import torch
import torch.utils.data
import pandas as pd
import numpy as np
import os


def loadImage(path: str):
    image = np.expand_dims(np.load(path), 0)
    image = np.log(image, where=image > 1e-10)
    image = np.nan_to_num(image)
    image = torch.from_numpy(image).double()
    return image


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
        image = loadImage(img_name)
        poses = self.sample_frame.iloc[idx, 2:]
        poses = torch.tensor([poses]).squeeze(axis=0).double()

        return image, poses


class FoldDataset(object):
    def __init__(self, original_dataset: torch.utils.data.Dataset, indices: np.array):
        self.original = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original[self.indices[idx]]


def splitData(n: int, folds: int, shuffle: bool = True):
    x = np.array([i for i in range(n)], dtype=np.int64)
    if shuffle:
        np.random.shuffle(x)
    for i in range(folds):
        i_first = int(i / folds * n)
        i_next = int((i + 1) / folds * n)
        x_rest = np.append(x[:i_first], x[i_next:], axis=0)
        yield x_rest, x[i_first:i_next]


def splitKFold(dataset: torch.utils.data.Dataset, folds: int, shuffle: bool = True):
    n = len(dataset)
    for train_idx, val_idx in splitData(n, folds, shuffle=shuffle):
        yield FoldDataset(dataset, train_idx), FoldDataset(dataset, val_idx)


if __name__ == "__main__":
    dataset = PhaseDataset("data/set_2/samples", "data/set_2/data.csv")
    for train, val in splitKFold(dataset, 4):
        print(len(train), len(val))
