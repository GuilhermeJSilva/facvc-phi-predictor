import torch
import torch.nn as nn
import torch.optim as optim
from phipredictor.data_loader import PhaseDataset
from phipredictor.model import Net
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def fit(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.MSELoss,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
):
    model.train()
    for _ in range(epochs):

        loss_history = []
        for data in train_loader:
            inputs, truth = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
        return loss_history


if __name__ == "__main__":
    train_dataset = PhaseDataset("data/set_1/samples", "data/set_1/data.csv")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    model = Net().double()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    running_loss = fit(model, optimizer, criterion, 2, train_loader)

    plt.plot(running_loss)
    plt.show()
