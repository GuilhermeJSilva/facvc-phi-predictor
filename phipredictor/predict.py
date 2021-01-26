import torch
import torch.nn as nn
import numpy as np
import phipredictor.data_loader


def predict(model, target, index):
    m = torch.load(model)
    dataset = phipredictor.data_loader.PhaseDataset(
        target + "/samples", target + "/data.csv"
    )
    image, true_out = dataset[index]
    criterion = nn.MSELoss()
    out = m(image.unsqueeze(0))
    print(out)
    print(criterion(out, true_out.unsqueeze(0)))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    predict(**vars(args))