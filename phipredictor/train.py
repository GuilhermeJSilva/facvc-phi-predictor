import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from phipredictor.data_loader import PhaseDataset
from phipredictor.model import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def validate(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, criteria: nn.MSELoss
):
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


if __name__ == "__main__":
    batch_size = 4
    train_dataset = PhaseDataset("data/set_1/samples", "data/set_1/data.csv")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    val_dataset = PhaseDataset("data/set_2/samples", "data/set_2/data.csv")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = Net().double()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    epochs = 10
    loss_hist, val_loss_hist = fit(model, optimizer, criterion, epochs, train_loader, val_loader)

    plt.plot(loss_hist)
    plt.plot([i * len(train_dataset) / batch_size for i in range(epochs + 1)], val_loss_hist)
    plt.show()
