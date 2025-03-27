"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
from model.utils import dist_util, logger

# TODO: load dataset
from data.load_data import load_data  # from diffuseq.text_datasets import load_data_text


from model.diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
)
from train import TrainLoop
from transformers import set_seed
import wandb

from model.tokeinzer import load_tokenizer
from model.encoder import LSTMAutoencoder

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight,
        verbose=True
    )
    next(data)

    data_valid = load_data(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='val',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight,
        verbose=True
    )

    lstm_encoder = LSTMAutoencoder(
        input_dim=2,  # Assuming 2D coordinates
        hidden_dim=args.hidden_dim,
        num_layers=2,  # if you change this, also update the matching coord_encoder LSTM in the transformer
    )

    print("Training LSTM Autoencoder...")
    # unsupervised pretrain the encoder
    for i in range(1000):
        batch = next(data)
        lstm_encoder.train_batch(batch)
    print("LSTM Autoencoder training complete.")

    lstm_encoder.to(dist_util.dev())

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    model.coord_encoder.load_state_dict(lstm_encoder.encoder.state_dict())

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()


if __name__ == "__main__":
    main()
