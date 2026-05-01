from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class IrisDataset(Dataset):
    """PyTorch dataset for normalized 128x128 grayscale iris images."""

    def __init__(self, X_images, y_labels):
        self.images = torch.as_tensor(X_images, dtype=torch.float32)
        self.labels = torch.as_tensor(y_labels, dtype=torch.long)

        if self.images.ndim != 3:
            raise ValueError("X_images must have shape (N, 128, 128)")

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("X_images and y_labels must contain the same number of samples")

        self.images = self.images.unsqueeze(1)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class IrisCNN(nn.Module):
    """Simple convolutional neural network for iris classification."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def _accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0

    return correct / total


def train_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    model_path="models/cnn_model.pth",
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """Train an IrisCNN on CPU and save the best model by validation accuracy."""
    device = torch.device("cpu")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = IrisDataset(X_train, y_train)
    val_dataset = IrisDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = IrisCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_accuracy = _accuracy(model, val_loader, device)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_state, model_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_cnn(model, X_test, y_test, batch_size=32):
    """Evaluate a CNN on CPU and return true and predicted labels as arrays."""
    device = torch.device("cpu")
    test_dataset = IrisDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions)

    return np.asarray(y_true), np.asarray(y_pred)
