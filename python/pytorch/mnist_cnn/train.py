from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = CNN(n_classes=10)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_fn=nn.CrossEntropyLoss()
    )
    trainer.train(train_dataloader, test_dataloader)


class CNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=int(32 * (28 / 4) * (28 / 4)), out_features=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


@dataclass
class Trainer():
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    max_step: int = 500
    steps_per_eval: int = 100

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        for step in range(self.max_step):
            if step % self.steps_per_eval == 0:
                self.print(step, train_dataloader, val_dataloader)
            xb, yb = next(iter(train_dataloader))
            loss = self.loss_fn(self.model(xb), yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def print(self, step: int, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        train_loss = self.loss(train_dataloader)
        val_loss = self.loss(val_dataloader)
        print(f"Step {step}: train={round(train_loss, 6):.6f} val={round(val_loss, 6):.6f}")

    @torch.no_grad()
    def loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        losses = []
        for x, y in dataloader:
            losses.append(self.loss_fn(self.model(x), y))
        self.model.train()
        return float(torch.mean(torch.Tensor(losses)))



main()
