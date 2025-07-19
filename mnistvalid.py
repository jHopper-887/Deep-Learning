import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import optuna as op

# prompt: create a vaildation set from this trainset

# define training and validation dataloaders
num_workers = 0
# percentage of training set to use as validation
valid_size = 0.2

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(
    root="~/.pytorch/MNIST_data/", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="~/.pytorch/MNIST_data/", train=False, download=True, transform=transform
)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))

np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# load training and validation data in batches
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, sampler=train_sampler, num_workers=num_workers
)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, sampler=valid_sampler, num_workers=num_workers
)

# load test data in batches
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, num_workers=num_workers
)


class Mnist(nn.Module):
    def __init__(self, xe):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(xe),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(xe),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layer(x)


device = "cpu"
model = Mnist(0.4277521083661638)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0006655603881611906)

epochs = 20
valid_min = np.inf


# def objective(trial):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ])
#     data_train = datasets.MNIST(
#         "MNIST_dat", download=True, train=True, transform=transform
#     )
#     data_test = datasets.MNIST(
#         "MNIST_dat", download=True, train=False, transform=transform
#     )
#     loader = DataLoader(data_train, shuffle=True, batch_size=64)
#     test_loader = DataLoader(data_test, shuffle=False, batch_size=64)
#     dropout = trial.suggest_float("dropout", 0, 0.5)
#     model = Mnist(xe=dropout)
#     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

#     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
#     loss = nn.CrossEntropyLoss()
#     for e in range(5):
#         for images, label in loader:
#             model.train()
#             output = model(images)
#             lo = loss(output, label)
#             optimizer.zero_grad()
#             lo.backward()
#             optimizer.step()

#     correct = 0
#     total = 0
#     with torch.no_grad():
#         model.eval()
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return total - correct


# study = op.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
# print("Best Accuracy:", study.best_value)
# print("Best Hyperparameters:", study.best_params)


for e in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    for images, label in train_loader:
        model.train()
        label = label.to(device)
        images = images.to(device)
        output = model(images)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    for images, label in valid_loader:
        model.eval()
        images = images.to(device)
        label = label.to(device)
        output = model(images)
        loss = loss_fn(output, label)
        valid_loss += loss.item()
    print(f"Train_loss: {train_loss}, Valid_loss: {valid_loss}")
    if valid_loss <= valid_min:
        print(
            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                valid_min, valid_loss
            )
        )
        valid_min = valid_loss
        torch.save(model.state_dict(), "model.pth")

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total}")
