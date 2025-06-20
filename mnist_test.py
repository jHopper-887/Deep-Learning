import torch
from torchvision import datasets, transforms
import torch.nn as nn


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


model = torch.load("best_mnist.pth", weights_only=False)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

testset = datasets.MNIST("Mnist_test", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=64)

images, labels = next(iter(testloader))

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
