import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


def train_val_dataloaders(train_data, test_data):
    print(f"Using device: {config.DEVICE}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation (test) data size: {len(test_data)}")

    if not train_data:
        raise ValueError("train_data list is empty.")
    perform_validation = bool(test_data)
    if not perform_validation:
        print("Warning: test_data is empty. Skipping validation.")

    x_mean, x_std, y_mean, y_std = normalization(train_data)

    # Create Datasets
    train_dataset = SymbolicRegressionDataset(train_data, x_mean, x_std, y_mean, y_std)
    val_loader = None
    if perform_validation:
        val_dataset = SymbolicRegressionDataset(test_data, x_mean, x_std, y_mean, y_std)
        val_loader = DataLoader(val_dataset, batch_size=config.VALIDATION_BATCH_SIZE, shuffle=False, num_workers=0,
                                pin_memory=True if config.DEVICE == "cuda" else False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=True if config.DEVICE == "cuda" else False)

    return train_loader, val_loader


def load_diffusion_data(input_dir="diffusion_data", file_name="train.json"):
    """Load processed diffusion data from the specified JSON file."""
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {str(e)}")

    print(f"Loaded {len(data)} records from {file_path}")
    return data


def normalization(train_data):
    print("Calculating normalization statistics from train_data...")
    all_coords_list_gen = [item['X_Y_combined'] for item in train_data]
    all_coords_np_gen = np.array(all_coords_list_gen, dtype=np.float32)
    all_coords_np_gen = np.nan_to_num(all_coords_np_gen, nan=0.0, posinf=0.0, neginf=0.0)
    x_mean_gen = np.mean(all_coords_np_gen[:, :, 0])
    x_std_gen = np.std(all_coords_np_gen[:, :, 0])
    y_mean_gen = np.mean(all_coords_np_gen[:, :, 1])
    y_std_gen = np.std(all_coords_np_gen[:, :, 1])
    x_std_gen = x_std_gen if x_std_gen > 1e-6 else 1.0
    y_std_gen = y_std_gen if y_std_gen > 1e-6 else 1.0
    print(
        f"Using Normalization Stats: X ~ N({x_mean_gen:.3f}, {x_std_gen:.3f}^2), Y ~ N({y_mean_gen:.3f}, {y_std_gen:.3f}^2)")
    return x_mean_gen, x_std_gen, y_mean_gen, y_std_gen


def load_data(input_dir="diffusion_data", version=7):
    train_data = load_diffusion_data(input_dir=input_dir, file_name=f"trainv{version}.json")
    test_data = load_diffusion_data(input_dir=input_dir, file_name=f"testv{version}.json")
    print(f"First train record: {train_data[0]}")
    print(f"First test record: {test_data[0]}")

    return train_data, test_data


class SymbolicRegressionDataset(Dataset):
    def __init__(self, data, x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=1.0):
        self.data = data
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
        self.processed_data = []
        for item in data:
             token_ids = np.array(item['token_ids'], dtype=np.int64)
             if np.any(token_ids >= config.VOCAB_SIZE):
                 token_ids = np.clip(token_ids, 0, config.VOCAB_SIZE - 1)

             xy_coords = np.array(item['X_Y_combined'], dtype=np.float32)
             xy_coords[:, 0] = (xy_coords[:, 0] - self.x_mean) / (self.x_std + 1e-8)
             xy_coords[:, 1] = (xy_coords[:, 1] - self.y_mean) / (self.y_std + 1e-8)
             condition_tensor = torch.from_numpy(xy_coords)

             self.processed_data.append({
                 'token_ids': torch.from_numpy(token_ids),
                 'condition': condition_tensor
             })

    def __len__(self): # Correct ':'
        return len(self.processed_data)

    def __getitem__(self, idx): # Correct ':'
        return self.processed_data[idx]