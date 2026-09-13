# Transfer learning
# Data augmentation
# Pytorch

"""
How does augmentation strength affect
the generalization of a CNN trained
on a small CIFAR-10 dataset?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 4 Experiments
#   No augmentation
#   Moderate Augmentation
#   Stronger Augmentation
#   Bad Augmentation

# Generalization gap = training accuracy - validation accuracy

# HYPER-PARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3


def train_no_aug_loader():
    pass


def train_mod_aug_loader():
    pass


def train_strong_aug_loader():
    pass


def train_bad_aug_loader():
    pass


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset = datasets.ImageFolder(root="data/train", transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 3 channels (RGB), Output: 16 feature maps, Kernel size: 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 16 channels, Output: 32 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # Fully Connected layers (Image is 8x8 after two poolings of a 32x32 image)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        # Apply relu to the output of the connected layer1
        x = F.relu(self.fc1(x))

        # Apply relu to the output of the connected layer2
        x = F.relu(self.fc2(x))

        logits = self.fc3(x)
        return logits


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model():
    for epoch in range(EPOCHS):
        print(f"---------------Running epoch-----{epoch+1}")
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the configured device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients from previous step
            loss.backward()
            optimizer.step()


def test_pipeline():
    print()
    pass


if __name__ == "__main__":
    train_model()
