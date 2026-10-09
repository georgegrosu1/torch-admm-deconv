"""
PyTorch implementation of a simple DnCNN model.
 
Original TensorFlow/Keras version:
Created on Fri May 16 09:13:27 2025
 
@author: Romulus Terebes
"""
 
import argparse
import os
import random
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train the DnCNN denoising model."
)
parser.add_argument(
    "-m",
    "--mode",
    choices=("cpu", "gpu"),
    default="gpu" if torch.cuda.is_available() else "cpu",
    help="Training device (default: automatically select gpu when available)."
)
args = parser.parse_args()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
 
SEED = 42
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
 
 
# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------
 
if args.mode == "gpu":
    if not torch.cuda.is_available():
        parser.error("GPU mode was requested, but CUDA is not available.")

    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 
print(f"Using device: {device}")
 
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
 
 
# ---------------------------------------------------------------------
# DnCNN model
# ---------------------------------------------------------------------
 
class DnCNN(nn.Module):
    """
    Simplified DnCNN model.
 
    The convolutional network predicts the noise component.
    The final denoised image is obtained as:
 
        denoised = noisy_input - predicted_noise
    """
 
    def __init__(self, depth=5, filters=64, image_channels=1):
        super().__init__()
 
        if depth < 3:
            raise ValueError("The DnCNN depth must be at least 3.")
 
        layers = []
 
        # First layer: Conv + ReLU, without Batch Normalization
        layers.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )
        layers.append(nn.ReLU(inplace=True))
 
        # Intermediate layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=3,
                    padding=1,
                    bias=True
                )
            )
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))
 
        # Final layer: estimated noise
        layers.append(
            nn.Conv2d(
                in_channels=filters,
                out_channels=image_channels,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )
 
        self.noise_estimator = nn.Sequential(*layers)
 
    def forward(self, noisy_image):
        estimated_noise = self.noise_estimator(noisy_image)
        denoised_image = noisy_image - estimated_noise
 
        return denoised_image
 
 
# ---------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------
 
def print_model_summary(model):
    print(model)
 
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
 
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
 
    print(f"\nTotal parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
 
 
# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------
 
def prepare_images(images):
    """
    Convert images to float32, normalize them to [0, 1],
    and convert them to PyTorch NCHW format.
 
    Supported input shapes:
        (N, H, W)
        (N, H, W, 1)
        (N, 1, H, W)
    """
 
    images = np.asarray(images, dtype=np.float32)
 
    # Normalize only if the data appears to be stored in [0, 255].
    if images.max() > 1.0:
        images = images / 255.0
 
    if images.ndim == 3:
        # NumPy: (N, H, W)
        # PyTorch: (N, 1, H, W)
        images = np.expand_dims(images, axis=1)
 
    elif images.ndim == 4:
        if images.shape[-1] == 1:
            # Convert NHWC to NCHW.
            images = np.transpose(images, (0, 3, 1, 2))
 
        elif images.shape[1] != 1:
            raise ValueError(
                f"Unsupported four-dimensional image shape: {images.shape}"
            )
 
    else:
        raise ValueError(
            f"Expected a three- or four-dimensional array, got {images.shape}"
        )
 
    images = np.ascontiguousarray(images)
 
    return torch.from_numpy(images)
 
 
train_data = np.load("train_patches.npz")
 
print("Training arrays:", train_data.files)
 
x_train = train_data["x_train"][:20000]
x_train_noisy = train_data["y_train"][:20000]
 
train_data.close()
 
 
test_data = np.load("test_patches.npz")
 
print("Testing arrays:", test_data.files)
 
x_test = test_data["x_test"][:5000]
x_test_noisy = test_data["y_test"][:5000]
 
test_data.close()
 
 
# Clean images are the targets.
x_train = prepare_images(x_train)
x_test = prepare_images(x_test)
 
# Noisy images are the inputs.
x_train_noisy = prepare_images(x_train_noisy)
x_test_noisy = prepare_images(x_test_noisy)
 
print(f"Training clean shape: {x_train.shape}")
print(f"Training noisy shape: {x_train_noisy.shape}")
print(f"Testing clean shape: {x_test.shape}")
print(f"Testing noisy shape: {x_test_noisy.shape}")
 
 
# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
 
BATCH_SIZE = 128
 
train_dataset = TensorDataset(x_train_noisy, x_train)
validation_dataset = TensorDataset(x_test_noisy, x_test)
 
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda"
)
 
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda"
)
 
 
# ---------------------------------------------------------------------
# Create model, loss, and optimizer
# ---------------------------------------------------------------------
 
model = DnCNN(
    depth=17,
    filters=64,
    image_channels=1
).to(device)
 
print_model_summary(model)
 
criterion = nn.MSELoss()
 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
 
 
# ---------------------------------------------------------------------
# Training and validation functions
# ---------------------------------------------------------------------
 
def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
 
    total_loss = 0.0
    total_samples = 0
 
    for noisy_images, clean_images in data_loader:
        noisy_images = noisy_images.to(
            device,
            non_blocking=True
        )
 
        clean_images = clean_images.to(
            device,
            non_blocking=True
        )
 
        optimizer.zero_grad(set_to_none=True)
 
        denoised_images = model(noisy_images)
        loss = criterion(denoised_images, clean_images)
 
        loss.backward()
        optimizer.step()
 
        batch_size = noisy_images.size(0)
 
        total_loss += loss.item() * batch_size
        total_samples += batch_size
 
    return total_loss / total_samples
 
 
@torch.no_grad()
def validate(model, data_loader, criterion, device):
    model.eval()
 
    total_loss = 0.0
    total_samples = 0
 
    for noisy_images, clean_images in data_loader:
        noisy_images = noisy_images.to(
            device,
            non_blocking=True
        )
 
        clean_images = clean_images.to(
            device,
            non_blocking=True
        )
 
        denoised_images = model(noisy_images)
        loss = criterion(denoised_images, clean_images)
 
        batch_size = noisy_images.size(0)
 
        total_loss += loss.item() * batch_size
        total_samples += batch_size
 
    return total_loss / total_samples
 
 
# ---------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------
 
EPOCHS = 10
 
history = {
    "loss": [],
    "val_loss": []
}
 
best_validation_loss = float("inf")
 
for epoch in range(EPOCHS):
    training_loss = train_one_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
 
    validation_loss = validate(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        device=device
    )
 
    history["loss"].append(training_loss)
    history["val_loss"].append(validation_loss)
 
    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
        f"loss: {training_loss:.8f} | "
        f"val_loss: {validation_loss:.8f}"
    )
 
    # Save the best model according to validation loss.
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
 
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "depth": 5,
                "filters": 64,
                "image_channels": 1
            },
            "simple_model_d3_best.pth"
        )
 
 
# ---------------------------------------------------------------------
# Save training history
# ---------------------------------------------------------------------
 
history_df = pd.DataFrame(history)
history_df.index.name = "epoch"
 
# Start epoch numbering at 1 instead of 0.
history_df.index = history_df.index + 1
 
history_df.to_csv("cnn_training_history.csv")
 
print("\nTraining history saved to cnn_training_history.csv")
 
 
# ---------------------------------------------------------------------
# Test model on one image
# ---------------------------------------------------------------------
 
model.eval()
 
sample_image = x_test_noisy[0:1]
 
with torch.no_grad():
    denoised_image = model(
        sample_image.to(device)
    )
 
denoised_image = denoised_image.cpu().numpy()
 
# Convert from NCHW to a two-dimensional image.
noisy_display = sample_image[0, 0].numpy()
original_display = x_test[0, 0].numpy()
denoised_display = denoised_image[0, 0]
 
# The residual operation may produce values slightly outside [0, 1].
denoised_display = np.clip(denoised_display, 0.0, 1.0)
 
 
# ---------------------------------------------------------------------
# Visualize results
# ---------------------------------------------------------------------
 
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 3, 1)
plt.imshow(noisy_display, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("Noisy Image")
plt.axis("off")
 
plt.subplot(1, 3, 2)
plt.imshow(original_display, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("Original Image")
plt.axis("off")
 
plt.subplot(1, 3, 3)
plt.imshow(denoised_display, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("Denoised Image")
plt.axis("off")
 
plt.tight_layout()
plt.show()
 
 
# ---------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------
 
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "depth": 5,
        "filters": 64,
        "image_channels": 1
    },
    "simple_model_d3.pth"
)
 
print("Final model saved to simple_model_d3.pth")
print("Best model saved to simple_model_d3_best.pth")