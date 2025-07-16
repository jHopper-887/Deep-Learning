import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import optuna

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(
    "cifar", download=True, train=True, transform=train_transform
)
testset = datasets.CIFAR10(
    "cifar", download=True, train=False, transform=test_transform
)

num_workers = 10
batch_size = 16
valid_size = 0.2

length = len(trainset)
indices = list(range(length))
np.random.shuffle(indices)
split = int(length * valid_size)
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(
    trainset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler
)
validloader = DataLoader(
    trainset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler
)
testloader = DataLoader(
    testset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)


class Cifar_Classifier(nn.Module):
    def __init__(self):
        super(Cifar_Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(20, 40, 3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(40, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.conv(x)


model = Cifar_Classifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003051308607576353)


# def objective(trial):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     train_dataset = datasets.CIFAR10(root="cifar", train=True, transform=transform)
#     train_data, val_data = random_split(train_dataset, [40000, 10000])
#     train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

#     model = Cifar_Classifier()

#     optimizer_name = trial.suggest_categorical(
#         "optimizer", ["Adam", "RMSprop", "AdamW", "SGD"]
#     )
#     lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
#     optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(5):
#         model.train()
#         for batch in train_loader:
#             inputs, targets = batch[0], batch[1]
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#     # Validate
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             inputs, targets = batch[0], batch[1]
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()

#     accuracy = correct / total
#     return accuracy


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)  # You can increase n_trials

# print("Best Accuracy:", study.best_value)
# print("Best Hyperparameters:", study.best_params)


def train(model, optim, loss_fn, trainloader, validloader, epochs):
    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        for image, label in trainloader:
            output = model(image)
            loss = loss_fn(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        curr_loss = 0
        count = 0
        with torch.no_grad():
            for image, label in validloader:
                output = model(image)
                loss = loss_fn(output, label)
                curr_loss += loss.item()
                count += 1
        curr_loss /= count
        if curr_loss < best_loss:
            print(f"Loss: {best_loss} --> {curr_loss}")
            torch.save(model.state_dict(), "cifar_10.pth")
            print("Saving Model....")
            best_loss = curr_loss
        print("Model Saved! :)")


train(model, optimizer, loss_fn, trainloader, validloader, 8)

model.load_state_dict(torch.load("cifar_10.pth"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for image, label in testloader:
        output = model(image)
        _, pred = torch.max(output, 1)
        correct += (pred == label).sum().item()
        total += label.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")
