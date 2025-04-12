import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from model.utils import dist_util, logger
from data.load_data import load_data
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
)
from model.diffuseq.step_sample import create_named_schedule_sampler
from model.tokeinzer import load_tokenizer


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())

    # Override some defaults for evaluation
    defaults.update({
        "batch_size": 4,
        "split": "test",
    })

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # Add only evaluation-specific arguments that aren't in defaults
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default=None, help="File to save generations")

    return parser


def load_checkpoint(checkpoint_path):
    logger.log(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=dist_util.dev())
    return checkpoint


def generate_samples(model, diffusion, batch, args):
    """Generate samples from the model using the diffusion process."""
    model.eval()

    with torch.no_grad():
        # Handle the batch whether it's a dict or list
        if isinstance(batch, list):
            # If batch is a list, assume no conditioning is needed
            cond = {}
        elif isinstance(batch, dict):
            # If batch is a dict, extract conditioning info
            cond = batch.get("cond", {})
        else:
            # Fallback
            cond = {}

        # Get model input dimensions
        input_dims = model.input_dims if hasattr(model, 'input_dims') else args.hidden_size

        # Set defaults for missing arguments
        clip_denoised = getattr(args, 'clip_denoised', True)
        clamp_step = getattr(args, 'clamp_step', 0)  # Default to 0 if not specified
        clamp_first = getattr(args, 'clamp_first', True)

        # Generate samples
        samples = diffusion.p_sample_loop(
            model,
            (args.batch_size, args.seq_len, input_dims),
            clip_denoised=clip_denoised,
            progress=True,
            model_kwargs=cond,
            clamp_step=clamp_step,
            clamp_first=clamp_first
        )

    return samples


def evaluate(args):
    """Evaluate the model on the specified dataset."""
    dist_util.setup_dist()
    logger.configure()

    logger.log("### Loading tokenizer and model embeddings...")
    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    logger.log("### Loading evaluation data...")
    eval_data = load_data(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split="train",
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.to(dist_util.dev())

    checkpoint = load_checkpoint(args.model_path)
    model.load_state_dict(checkpoint)

    for _ in range(10):
        batch = next(eval_data)
        samples = generate_samples(model, diffusion, batch, args)

        prd_tokens = tokenizer.decode(model.get_logits(samples[-1]).argmax(dim=-1).tolist()[0])
        grt_tokens = tokenizer.decode(batch[1]['input_ids'].tolist()[0])

        print('\n')
        print("Generated:", prd_tokens)
        print("Ground truth:", grt_tokens)
        print('\n')


def main():
    args = create_argparser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
