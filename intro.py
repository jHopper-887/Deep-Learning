import torch
from torchvision import datasets, transforms
# import intel_extension_for_pytorch as ipex

# import torch.nn.functional as F
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

trainset = datasets.MNIST(
    "Documents/Pytorch/MNIST", download=True, train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)


class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


model = Mnist()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss = torch.nn.CrossEntropyLoss()
# model, optimizer = ipex.optimize(model=model, optimizer=optimizer, dtype=torch.float)


def train(model, train_loader, optimizer, loss_fn, epochs, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed with total loss: {total_loss:.4f}")


train(model, trainloader, optimizer, loss, 4)

model.eval()

sample_image = images[1].unsqueeze(0)
true_label = labels[1].item()

with torch.no_grad():
    output = model(sample_image)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"True Label: {true_label}")
print(f"Predicted Digit: {predicted_class}")
