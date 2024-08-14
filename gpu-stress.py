import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"Using {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Exiting...")
    raise SystemExit


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel().to(device)

# Create random input data
input_data = torch.randn(64, 3, 128, 128).to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting GPU stress test...")

# Run an infinite loop to stress the GPU
while True:
    optimizer.zero_grad()
    output = model(input_data)
    target = torch.randint(0, 10, (64,)).to(device)  # Random target
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")
