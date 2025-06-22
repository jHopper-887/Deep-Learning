import torch
from torchvision import datasets, transforms
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.LogSoftmax(dim=1),
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

trainset = datasets.FashionMNIST(
    "Fashion_Mnist", download=True, train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(model, loader, loss_fn, optim, epochs):
    for e in range(epochs):
        model.train()
        net_loss = 0
        for i, (images, label) in enumerate(loader):
            output = model.forward(images)
            loss = loss_fn(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            net_loss += loss

            if (i + 1) % 100 == 0:
                print(f"Epoch {e + 1}, Step {i + 1}, Loss {loss}")
        print(f"    The net loss is {net_loss}")


train(model, trainloader, loss_fn, optimizer, 22)

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in trainloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

torch.save(model, "fshn_mnist")
