import torch
from torchvision import datasets, transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

trainset = datasets.MNIST("MNIST_dat", transform=transform, download=True, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 28 * 28),
            nn.LeakyReLU(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


model = Mnist()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(model, loader, loss_fn, optim, epochs):
    for epoch in range(epochs):
        model.train()
        net_loss = 0
        i = 0
        for images, label in loader:
            loss = loss_fn(model(images), label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            net_loss += loss
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch}, Step {i + 1}, Loss {loss}")
            i += 1
        print(f"Total Loss for Epoch {epoch}: {net_loss}")


train(model, trainloader, loss_fn, optim, 8)

torch.save(model, "best_mnist.pth")
