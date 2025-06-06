import numpy as np
import os
from pathlib import Path
import torch

from Spectrogram import Spectrogram


# Only need to run once. Separates files in training and validation into directories based on their instruments.
def separate_data_into_directories(directory: str):
    """
    Separates NSynth dataset into separate directories based on their instrument.
    Prints out how many files are in each intstrument directory.

    --- inputs ---------------- 
    type_:str = 'train' / 'validation' / 'test' depending on type of data
    """
    data_path = Path('data/')
    train_test_dir = data_path / directory
    class_names = ['bass', 'brass', 'flute', 
                   'guitar', 'keyboard', 'mallet', 
                   'organ', 'reed', 'string', 
                   'synth', 'vocal']
    
    print('--- Creating Directories ----------------')
    for class_ in class_names:
        class_dir = train_test_dir / class_
        if class_dir.is_dir():
            print(f'{class_dir} already exists.')
        else:
            print(f'Creating {class_dir} directory...')
            class_dir.mkdir(parents=True, exist_ok=True)

    print('\n--- Moving Files ----------------')
    for entry in (data_path/directory).iterdir():
        if entry.is_file() and entry.suffix == '.wav':
            print(f'{entry.name} moved')
            os.rename(entry, train_test_dir/entry.name.split('_')[0]/entry.name)

    print('\n--- Length of Directories ----------------')
    for subdir in train_test_dir.iterdir():
        if subdir.is_dir():
            num_files = len([f for f in subdir.iterdir() if f.is_file()])
            print(f"{subdir.name}: {num_files} files")


def audio_to_spectrograms():
    """
    Walks every WAV file under:
        data/train_audio/<instrument>/*.wav
        data/validation_audio/<instrument>/*.wav
        data/test_audio/<instrument>/*.wav

    and saves a matching PNG under:
        data/train_image/<instrument>/*.png
        data/validation_image/<instrument>/*.png
        data/test_image/<instrument>/*.png

    If a PNG already exists, it is skipped.
    """
    train_audio_dir = Path("data/train_audio")
    val_audio_dir   = Path("data/validation_audio")
    test_audio_dir  = Path("data/test_audio")

    train_image_dir = Path("data/train")
    val_image_dir   = Path("data/validation")
    test_image_dir  = Path("data/test")

    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    test_image_dir.mkdir(parents=True, exist_ok=True)

    train_audio_paths = sorted(train_audio_dir.glob("*/*.wav"))
    val_audio_paths   = sorted(val_audio_dir.glob("*/*.wav"))
    test_audio_paths  = sorted(test_audio_dir.glob("*/*.wav"))

    # --- PROCESS TRAINING SET ---
    print("=== TRAINING SET ===")
    for audio_path in train_audio_paths:
        instrument = audio_path.stem.split("_")[0]
        image_name = audio_path.stem + ".png"
        image_path = train_image_dir / instrument / image_name

        image_path.parent.mkdir(parents=True, exist_ok=True)

        if image_path.exists():
            print(f"[SKIP]    {image_path.name} already exists")
            continue

        print(f"[GENERATE] {audio_path.name} → {image_path.name}")
        spec = Spectrogram(file_path=audio_path)
        spec.save_spec(train_image_dir)

    # --- PROCESS VALIDATION SET ---
    print("\n=== VALIDATION SET ===")
    for audio_path in val_audio_paths:
        instrument = audio_path.stem.split("_")[0]
        image_name = audio_path.stem + ".png"
        image_path = val_image_dir / instrument / image_name

        image_path.parent.mkdir(parents=True, exist_ok=True)

        if image_path.exists():
            print(f"[SKIP]    {image_path.name} already exists")
            continue

        print(f"[GENERATE] {audio_path.name} → {image_path.name}")
        spec = Spectrogram(file_path=audio_path)
        spec.save_spec(val_image_dir)

    # --- PROCESS TEST SET ---
    print("\n=== TEST SET ===")
    for audio_path in test_audio_paths:
        instrument = audio_path.stem.split("_")[0]
        image_name = audio_path.stem + ".png"
        image_path = test_image_dir / instrument / image_name

        image_path.parent.mkdir(parents=True, exist_ok=True)

        if image_path.exists():
            print(f"[SKIP]    {image_path.name} already exists")
            continue

        print(f"[GENERATE] {audio_path.name} → {image_path.name}")
        spec = Spectrogram(file_path=audio_path)
        spec.save_spec(test_image_dir)


def train_step(model, dataloader, criterion, optimiser, device):
    model.train()
    model.to(device)

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    # --- LOOP OVER ALL BATCHES ---
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 1) Forward pass
        y_logits = model(X)
        loss = criterion(y_logits, y)

        # 2) Backward + optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # 3) Accumulate statistics
        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        preds = y_logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

    # Compute epoch‐level averages
    train_loss = running_loss / total_samples
    train_acc = total_correct / total_samples

    return train_loss, train_acc


def test_step(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():    
        # --- LOOP OVER ALL BATCHES ---
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

    test_loss = running_loss / total_samples
    test_acc = total_correct / total_samples

    return test_loss, test_acc


if __name__ == '__main__':
    # Run only one at a time if needed:
    # separate_data_into_directories(type_='train_audio')
    # separate_data_into_directories(type_='validation_audio')
    # separate_data_into_directories(type_='test_audio')
    # audio_to_spectrograms()
    pass