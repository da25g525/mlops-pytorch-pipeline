import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return total_loss / total, correct / total


def main():
    config_path = Path("/app/configs/training_config.yaml")

    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        config["model"]["architecture"],
        config["model"]["num_classes"]
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        config["data"]["data_dir"],
        config["training"]["batch_size"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_count = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4)
        }), flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0

            save_path = checkpoint_dir / config["output"]["model_name"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, save_path)

        else:
            patience_count += 1

            if patience_count >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch + 1
                }), flush=True)
                break


if __name__ == "__main__":
    main()
