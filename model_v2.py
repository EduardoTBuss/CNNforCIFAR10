import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from plot import plot_training  # type: ignore
except Exception:
    plot_training = None

try:
    from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY  # type: ignore
except Exception:
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4


class ParamCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        num_layers: int = 6,
        num_filters: Union[int, List[int]] = 64,
        kernel_size: Union[int, List[int]] = 3,
        downsample_every: Optional[int] = 2,
    ) -> None:
        super().__init__()
        assert num_layers >= 1

        if isinstance(num_filters, int):
            filters = [num_filters] * num_layers
        else:
            assert len(num_filters) == num_layers
            filters = list(num_filters)

        if isinstance(kernel_size, int):
            klist = [kernel_size] * num_layers
        else:
            assert len(kernel_size) == num_layers
            klist = list(kernel_size)

        layers: List[nn.Module] = []
        c_in = in_channels
        self.strides: List[int] = []

        for i in range(num_layers):
            k = int(klist[i])
            pad = k // 2
            stride = 2 if downsample_every and (i + 1) % downsample_every == 0 else 1
            self.strides.append(stride)

            conv = nn.Conv2d(c_in, filters[i], kernel_size=k, stride=stride, padding=pad, bias=True)
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))
            c_in = filters[i]

        self.features = nn.Sequential(*layers)

        h = w = 32
        for s in self.strides:
            if s == 2:
                h = (h + 1) // 2
                w = (w + 1) // 2
        flattened = c_in * h * w
        self.classifier = nn.Linear(flattened, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@torch.inference_mode()
def evaluate_model(model: nn.Module, dataloader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    max_epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    plateau_factor: float = 0.1,
    plateau_patience: int = 5,
    min_lr: float = 1e-6,
    es_patience: int = 10,
    es_delta: float = 1e-4,
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=es_delta,
        threshold_mode="rel",
        min_lr=min_lr,
        verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = 0.0
    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        improved = val_loss < (best_val_loss - es_delta)
        if improved:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if val_acc > best_acc:
            best_acc = val_acc

        if improved:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, "best_model.pth")

        if plot_training is not None:
            try:
                plot_training(history)
            except Exception:
                pass

        if no_improve >= es_patience:
            print(f"\nEarly stopping acionado após {epoch+1} épocas sem melhora da val_loss.")
            break

    print(f"\nMelhor acurácia de validação: {best_acc:.2f}% | Melhor val_loss: {best_val_loss:.4f}")
    return history, best_state


@torch.inference_mode()
def test_model(model: nn.Module, test_loader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Teste final | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    return test_loss, test_acc


# ---------------------------
# Exemplos de uso:
# ---------------------------
# 1) Kernel 3, 64 filtros, 6 camadas; downsample a cada 2 camadas
# model = ParamCNN(num_layers=6, num_filters=64, kernel_size=3, downsample_every=2)
#
# 2) Kernel variável e filtros por camada; downsample em todas as camadas (stride=2)
# model = ParamCNN(
#     num_layers=5,
#     num_filters=[32, 64, 128, 128, 128],
#     kernel_size=[3, 3, 5, 3, 3],
#     downsample_every=1,
# )
#
# 3) Kernel 5, 32 filtros, sem downsample (tamanho espacial preservado)
# model = ParamCNN(num_layers=4, num_filters=32, kernel_size=5, downsample_every=None)
#
# Para treino (ReduceLROnPlateau + EarlyStopping coordenados):
# history, best_state = train_model(
#     model,
#     train_loader,
#     val_loader,
#     device,
#     max_epochs=100,
#     lr=3e-4,
#     weight_decay=1e-4,
#     plateau_factor=0.1,
#     plateau_patience=5,
#     min_lr=1e-6,
#     es_patience=10,
#     es_delta=1e-4,
# )
# Para teste:
# test_model(model, test_loader, device)
