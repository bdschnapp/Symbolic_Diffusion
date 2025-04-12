from diffusion.discrete_diffusion import DiscreteDiffusion
from diffusion.d3pm import ConditionalD3PMTransformer

import os
import time
import config
import numpy as np
import torch


def load_model(path):
    loaded_model = ConditionalD3PMTransformer(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        seq_len=config.SEQ_LEN,
        condition_dim=config.XY_DIM,  # Pass XY_DIM here
        num_timesteps=config.NUM_TIMESTEPS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    if os.path.exists(path):
        print(f"Loading state dictionary from: {path}")
        try:
            state_dict = torch.load(path, map_location=config.DEVICE)
            # Load the state dictionary into the model instance
            loaded_model.load_state_dict(state_dict, strict=True)
            # Set to Evaluation Mode
            loaded_model.eval()
            print("Model weights loaded successfully and set to evaluation mode.")

        except Exception as e:
            print(f"ERROR: Failed to load state dictionary.")
            print(e)
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(
                    "\nArchitecture mismatch detected. Ensure the class definition above matches the saved model EXACTLY.")
            loaded_model = None
    else:
        print(f"ERROR: Model weights file not found at: {path}")
        loaded_model = None

    diffusion_helper = DiscreteDiffusion(
        num_timesteps=config.NUM_TIMESTEPS,
        vocab_size=config.VOCAB_SIZE,
        device=config.DEVICE
    )

    return loaded_model, diffusion_helper


@torch.no_grad()
def generate(data_item, trained_model, x_mean, x_std, y_mean, y_std):
    true_tokens = np.array(data_item['token_ids'])
    xy_coords_orig = np.array(data_item['X_Y_combined'], dtype=np.float32)

    # Normalize and Prepare Condition Tensor
    xy_coords_norm = np.copy(xy_coords_orig)
    xy_coords_norm[:, 0] = (xy_coords_norm[:, 0] - x_mean) / (x_std + 1e-8)
    xy_coords_norm[:, 1] = (xy_coords_norm[:, 1] - y_mean) / (y_std + 1e-8)
    condition_tensor = torch.from_numpy(xy_coords_norm).float()
    condition_tensor = condition_tensor.unsqueeze(0).to(config.DEVICE)  # (1, N_POINTS, XY_DIM)

    print(f"Ground Truth Tokens:\n{true_tokens}")
    print(f"\nCondition Tensor shape (Normalized): {condition_tensor.shape}")

    diffusion_sampler = DiscreteDiffusion(
        num_timesteps=config.NUM_TIMESTEPS,
        vocab_size=config.VOCAB_SIZE,
        device=config.DEVICE
    )

    trained_model.to(config.DEVICE)
    trained_model.eval()
    print("\nStarting generation process...")
    with torch.no_grad():
        generated_sample = diffusion_sampler.sample(
            model=trained_model,
            condition=condition_tensor,
            shape=(1, config.SEQ_LEN)
        )
    generated_tokens = generated_sample.cpu().numpy()[0]
    print(f"\nGenerated Tokens:\n{generated_tokens}")

    return generated_tokens


@torch.no_grad() # Ensure no gradients are calculated during generation
def generate_tokens(data_item, model, diffusion_helper,
                    x_mean, x_std, y_mean, y_std,
                    device=config.DEVICE, seq_len=config.SEQ_LEN,
                    n_points=config.N_POINTS, xy_dim=config.XY_DIM):
    """
    Generates a sequence of tokens for a single data item using the diffusion model
    (expects model with PointCloudEncoder).

    Args:
        data_item (dict): A dictionary containing at least 'X_Y_combined'.
        model (nn.Module): The loaded and trained ConditionalD3PMTransformer model.
        diffusion_helper (DiscreteDiffusion): An instantiated DiscreteDiffusion helper object.
        x_mean (float): Mean of X coordinates used for training normalization.
        x_std (float): Standard deviation of X coordinates used for training normalization.
        y_mean (float): Mean of Y coordinates used for training normalization.
        y_std (float): Standard deviation of Y coordinates used for training normalization.
        device (str): The device to run generation on ('cuda' or 'cpu').
        seq_len (int): The target sequence length.
        n_points (int): The number of X-Y points in the condition.
        xy_dim (int): The dimension of each point (should be 2).

    Returns:
        np.ndarray: A NumPy array of shape (seq_len,) containing the generated token IDs,
                    or None if an error occurs.
    """
    if not model:
        print("ERROR: Model not provided or not loaded.")
        return None
    if not diffusion_helper:
        print("ERROR: Diffusion helper not provided.")
        return None
    if 'X_Y_combined' not in data_item:
        print("ERROR: 'X_Y_combined' key missing from data_item.")
        return None

    # 1. Prepare the Condition Tensor (Corrected Shape)
    try:
        xy_coords = np.array(data_item['X_Y_combined'], dtype=np.float32)

        # Ensure input shape is correct before normalization
        if xy_coords.shape != (n_points, xy_dim):
             print(f"ERROR: Input xy_coords shape mismatch. Expected ({n_points}, {xy_dim}), got {xy_coords.shape}")
             return None

        # Apply the *exact same* normalization used during training
        xy_coords[:, 0] = (xy_coords[:, 0] - x_mean) / (x_std + 1e-8) # Add epsilon for safety
        xy_coords[:, 1] = (xy_coords[:, 1] - y_mean) / (y_std + 1e-8)

        # --- Convert to tensor WITHOUT flattening ---
        # condition_np = xy_coords.flatten() # REMOVED
        condition_tensor = torch.from_numpy(xy_coords).float() # Shape: (N_POINTS, XY_DIM)
        # ------------------------------------------

        # Add batch dimension and move to device
        condition_tensor = condition_tensor.unsqueeze(0).to(device) # Shape: (1, N_POINTS, XY_DIM)

        # --- Updated Shape Check ---
        if condition_tensor.shape != (1, n_points, xy_dim):
             print(f"ERROR: Final condition tensor shape mismatch. Expected (1, {n_points}, {xy_dim}), got {condition_tensor.shape}")
             return None
        # -------------------------

    except Exception as e:
        print(f"ERROR: Failed to preprocess condition: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

    # 2. Ensure model is in evaluation mode
    model.eval()

    # 3. Generate using the diffusion sampler
    print(f"Generating sequence for item (Condition shape: {condition_tensor.shape})...")
    start_time = time.time()
    try: # Add try-except around the sample call itself
        generated_sample_tensor = diffusion_helper.sample(
            model=model,
            condition=condition_tensor, # Pass the 3D tensor
            shape=(1, seq_len) # Generate 1 sequence of length seq_len
        )
    except Exception as e:
        print(f"ERROR: Exception during diffusion_helper.sample: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None if sampling fails

    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds.")

    # 4. Process the output
    # Move to CPU, remove batch dimension, convert to NumPy
    generated_tokens_np = generated_sample_tensor.cpu().squeeze(0).numpy()

    return generated_tokens_np