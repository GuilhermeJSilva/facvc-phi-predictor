import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from phipredictor.data_loader import PhaseDataset
from phipredictor.model import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_dataset = PhaseDataset("data/set_1/samples", "data/set_1/data.csv")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=False, num_workers=2
)

model = Net().double()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-12)


model.train()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, truth = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, truth)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")
