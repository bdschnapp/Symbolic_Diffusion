import config
import time
import sys
import os
import torch
import numpy as np

from generation import generate
from diffusion.discrete_diffusion import DiscreteDiffusion
from diffusion.d3pm import ConditionalD3PMTransformer

from data.load_data import train_val_dataloaders, load_data, normalization

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from IPython import display


@torch.no_grad()
def validation(model, diffusion, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        x_start = batch['token_ids'].to(device)
        condition = batch['condition'].to(device)
        if x_start.max() >= config.VOCAB_SIZE or x_start.min() < 0:
             print(f"\nWarning: Invalid token ID in validation batch. Skipping.")
             continue
        loss = diffusion.compute_loss(model, x_start, condition, pad_token_id=config.PAD_TOKEN_ID)
        if not torch.isnan(loss):
             total_val_loss += loss.item()
             num_batches += 1
        else:
            print("\nWarning: NaN loss encountered during validation. Skipping batch.")

    model.train()
    if num_batches == 0:
        print("\nWarning: No valid batches processed during evaluation.")
        return float('inf')
    return total_val_loss / num_batches


# --- Training Function ---
def train(train_loader, val_loader, perform_validation=True):
    # Initialize Model
    model = ConditionalD3PMTransformer(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        seq_len=config.SEQ_LEN,
        condition_dim=config.XY_DIM,
        num_timesteps=config.NUM_TIMESTEPS, dropout=config.DROPOUT
    ).to(config.DEVICE)

    diffusion = DiscreteDiffusion(num_timesteps=config.NUM_TIMESTEPS, vocab_size=config.VOCAB_SIZE, device=config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, config.PATIENCE // 2), verbose=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if perform_validation:
        print(f"Early stopping patience: {config.PATIENCE}")
        print(f"Best model will be saved to: {config.BEST_MODEL_PATH}")

    best_val_loss = float('inf'); epochs_no_improve = 0
    epochs_plotted = []; train_losses = []; val_losses = []

    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        processed_batches = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x_start = batch['token_ids'].to(config.DEVICE)
            condition = batch['condition'].to(config.DEVICE) # Shape (B, N_POINTS, XY_DIM)

            if x_start.max() >= config.VOCAB_SIZE or x_start.min() < 0:
                 print(f"\nWarning: Invalid token ID in train batch. Skipping.")
                 continue

            loss = diffusion.compute_loss(model, x_start, condition, pad_token_id=config.PAD_TOKEN_ID)

            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected during training. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            processed_batches += 1

            print(f"\rEpoch [{epoch+1}/{config.EPOCHS}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}   ", end="")
            sys.stdout.flush()

        # --- End of Epoch ---
        avg_train_loss = total_train_loss / processed_batches if processed_batches > 0 else 0
        epoch_time = time.time() - start_time

        # Validation Step
        avg_val_loss = float('inf')
        if perform_validation and val_loader:
            print(f"\nEpoch [{epoch+1}/{config.EPOCHS}] completed in {epoch_time:.2f}s. Avg Train Loss: {avg_train_loss:.4f}. Evaluating...", end="")
            sys.stdout.flush()
            avg_val_loss = validation(model, diffusion, val_loader, config.DEVICE)
            print(f" Avg Val Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)
        else:
             print(f"\nEpoch [{epoch+1}/{config.EPOCHS}] completed in {epoch_time:.2f}s. Avg Train Loss: {avg_train_loss:.4f}. (Validation Skipped)")

        # Store losses for plotting
        epochs_plotted.append(epoch + 1)
        train_losses.append(avg_train_loss)
        if perform_validation:
             val_losses.append(avg_val_loss)

        # Live Plotting
        try:
            display.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_plotted, train_losses, 'bo-', label='Training Loss')
            if val_losses:
                ax.plot(epochs_plotted, val_losses, 'ro-', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss Over Epochs')
            ax.grid(True)
            ax.legend()
            if max(train_losses, default=0) > 5 * min(train_losses, default=1):
                 ax.set_yscale('log')
            display.display(fig)
            plt.close(fig)
        except Exception as e:
            print(f"\nError during plotting: {e}")

        # Early Stopping Check
        if perform_validation:
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve from {best_val_loss:.4f}. Patience: {epochs_no_improve}/{config.PATIENCE}")
                if epochs_no_improve >= config.PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    # End of Training Loop
    print("Training finished.")
    if perform_validation and os.path.exists(config.BEST_MODEL_PATH):
        print(f"Loading best model weights from {config.BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    elif perform_validation:
        print("Warning: Best model path not found, but validation was performed. Returning model from last epoch.")
    else:
        print("Validation was not performed. Returning model from last epoch.")
    return model


def main():
    sample_idx = 42

    train_data, val_data = load_data(input_dir='data/Modified', version=7)

    x_mean, x_std, y_mean, y_std = normalization(train_data)

    train_loader, val_loader = train_val_dataloaders(train_data, val_data)

    trained_model = train(train_loader, val_loader)

    data_item_to_generate = val_data[sample_idx]

    generate(
        data_item=data_item_to_generate,
        trained_model=trained_model,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std
    )


if __name__ == "__main__":
    main()
