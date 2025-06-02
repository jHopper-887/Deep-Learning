import torch as t
import torch.nn as nn

# import matplotlib as plot
import intel_extension_for_pytorch as ipex


t.manual_seed(42)

weights = 0.006
bias = 0.23

X = t.arange(1, 45, 0.5).unsqueeze(dim=1)
y = X * weights + bias

train_split = int(0.6 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(t.randn(1, dtype=t.float), requires_grad=True)
        self.bias = nn.Parameter(t.randn(1, dtype=t.float), requires_grad=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.weights * x + self.bias


model = LinearRegression()


with t.inference_mode():
    predict = model(X_test)


print(f"The predicted values are: {predict}")
print(f"The exact values are: {y_test}")

loss_fn = nn.L1Loss()

optimizer = t.optim.Adam(params=model.parameters(), lr=0.0001)
model, optimizer = ipex.optimize(model=model, optimizer=optimizer)
epochs = 10000
step = 20

for epoch in range(epochs + step):
    # model.train() # setups the model for training
    y_pred = model(X_train)  # makes predictions using the model
    loss = loss_fn(y_pred, y_train)  # calculates the loss for the model
    optimizer.zero_grad()  # clears previous gradients from all the parameters.grad()
    loss.backward()  # calculates the gradient for the loss function
    optimizer.step()  # updates weights using current gradients
    # model.eval() # sets the model for evalution mode

    if epoch % step == 0:
        print(f"Epoch: {epoch} | L1Loss: {loss}")

print(
    f"The learned parameters are weight: {model.weights.item()}, bias: {model.bias.item()}"
)
print(f"The exact values are weight: {weights}, bias: {bias}")
