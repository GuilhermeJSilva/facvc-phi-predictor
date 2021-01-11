import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from phipredictor.data_loader import PhaseDataset, splitKFold
from phipredictor.model import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def validate(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, criteria: nn.MSELoss
):
    model.eval()
    losses = []
    for inputs, truth in val_loader:
        outputs = model(inputs)
        loss = criteria(outputs, truth)
        losses.append(loss.item())

    return np.mean(losses)


def fit(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criteria: nn.MSELoss,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
):
    model.train()
    loss_history = []
    val_loss_history = [validate(model, val_loader, criteria)]
    for _ in range(epochs):
        model.train()

        for data in train_loader:
            inputs, truth = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criteria(outputs, truth)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        val_loss_history.append(validate(model, val_loader, criteria))
    return loss_history, val_loss_history


def saveParameters(filename, parameters):
    with open(filename, "w") as f:
        json.dump(parameters, f)


def saveResults(filename, loss_hist, val_loss_hist, len_trainset, batch):
    with open(filename, "w") as f:
        f.write("Steps,Value,Validation\n")
        for i, v in enumerate(loss_hist):
            f.write(f"{i},{v},0\n")
            if i % 100 == 99:
                f.flush()
        for i, v in enumerate(val_loss_hist):
            f.write(f"{i * math.ceil(len_trainset / batch)},{v},1\n")
            if i % 100 == 99:
                f.flush()


def experiment(options):
    dataset = PhaseDataset(options.dataset + "/samples", options.dataset + "/data.csv")
    for i, (train_set, val_set) in enumerate(splitKFold(dataset, 4, shuffle=False)):
        os.makedirs(options.prefix, exist_ok=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=options.batch)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=options.batch)
        model = Net().double()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=options.lr)
        loss_hist, val_loss_hist = fit(
            model, optimizer, criterion, options.epochs, train_loader, val_loader
        )
        saveParameters(os.path.join(options.prefix, "parameters.json"), vars(options))
        saveResults(
            os.path.join(options.prefix, f"mean_error_f{i + 1}.csv"),
            loss_hist,
            val_loss_hist,
            len(train_set),
            options.batch,
        )
        torch.save(model, os.path.join(options.prefix, f"model_f{i+1}.pth"))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset to use")
    parser.add_argument("prefix", type=str, help="prefix to save the data")
    parser.add_argument(
        "-lr", type=float, dest="lr", help="learning rate", default=1e-5
    )
    parser.add_argument("-batch", type=int, dest="batch", help="batch size", default=4)
    parser.add_argument("-epochs", type=int, default=10, dest="epochs")
    args = parser.parse_args()

    experiment(args)
