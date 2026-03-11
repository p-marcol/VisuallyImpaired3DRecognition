import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import DummyDataset
import config

def train():
    device = config.DEVICE

    dataset = DummyDataset(
        length=200,
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.EPOCHS}, loss={epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "models/model_dummy.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()