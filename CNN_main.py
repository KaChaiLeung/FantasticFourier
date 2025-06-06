from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from NeuralNetworkV0 import NeuralNetworkV0
from NeuralNetworkV1 import NeuralNetworkV1
from functions import train_step, test_step


# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 3
NUM_WORKERS = 0 # Keep this as 0 to avoid issues, don't know why?

# Data transformation function
data_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    # Normalise for NNV1, otherwise don't
    # transforms.Normalize(
    #     mean = [0.485, 0.456, 0.406],
    #     std  = [0.229, 0.224, 0.225])
])

# Getting data
data_path = Path('data/')
train_dir = data_path / 'train'
val_dir = data_path / 'val'

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

val_data = datasets.ImageFolder(root=val_dir,
                                 transform=data_transform,
                                 target_transform=None)

classes = train_data.classes

# Dataloaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)

# Instance of NeuralNetworkV0
model_0 = NeuralNetworkV0(3, len(classes))

# Instance of NeuralNetworkV1
model_1 = NeuralNetworkV1(3, len(classes))

class_counts = Counter(train_data.targets)
class_freqs = torch.tensor([class_counts[i] for i in range(len(class_counts))], dtype=torch.float)
class_weights = 1.0 / class_freqs
class_weights = class_weights / class_weights.sum()

results = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
}

# Loss function and optimiser - NEED TO CHANGE DEPENDING ON MODEL

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimiser = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
# optimiser = torch.optim.SGD(model_1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

epoch_list = []

# Training and testing loop
for epoch in tqdm(range(EPOCHS)):

    train_loss, train_acc = train_step(model=model_0,
                                       dataloader=train_dataloader,
                                       criterion=criterion,
                                       optimiser=optimiser,
                                       device=device)

    val_loss, val_acc = test_step(model=model_0,
                                    dataloader=val_dataloader,
                                    criterion=criterion,
                                    device=device)

    print(f'{epoch} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['val_loss'].append(val_loss)
    results['val_acc'].append(val_acc)
    epoch_list.append(epoch)


plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(epoch_list, results['train_loss'], color='tab:blue', label='Train Loss')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right')

ax2 = plt.subplot(2, 2, 2)
ax2.plot(epoch_list, results['train_acc'], color='tab:red', label='Train Accuracy')
ax2.set_title("Training Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_ylim(bottom=0)
ax2.legend(loc='lower right')

ax3 = plt.subplot(2, 2, 3)
ax3.plot(epoch_list, results['val_loss'], color='tab:orange', label='Val Loss')
ax3.set_title("Validation Loss")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.set_ylim(bottom=0)
ax3.legend(loc='upper right')

ax4 = plt.subplot(2, 2, 4)
ax4.plot(epoch_list, results['val_acc'], color='tab:green', label='Val Accuracy')
ax4.set_title("Validation Accuracy")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy")
ax4.grid(True, linestyle='--', alpha=0.4)
ax4.set_ylim(bottom=0)
ax4.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Saving model
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'NNV0_3_Epochs'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)