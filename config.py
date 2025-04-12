import torch

# Data Parameters
SEQ_LEN = 12
N_POINTS = 30
XY_DIM = 2
PAD_TOKEN_ID = 0
VOCAB_SIZE = 17

# Model Capacity (Large)
EMBED_DIM = 192
NUM_HEADS = 12
NUM_LAYERS = 8
DIM_FEEDFORWARD = 768
DROPOUT = 0.1

# Diffusion Parameters
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
SCHEDULE_TYPE = 'cosine'

# Training Parameters
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 50
BEST_MODEL_PATH = "best_d3pm_pointnet_crossattn.pth"

EXP_ARG_MAX = 700


def set_parameters(args):
    global SEQ_LEN, N_POINTS, XY_DIM, PAD_TOKEN_ID, VOCAB_SIZE
    global EMBED_DIM, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT
    global NUM_TIMESTEPS, BETA_START, BETA_END, SCHEDULE_TYPE
    global BATCH_SIZE, VALIDATION_BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE, PATIENCE
    global BEST_MODEL_PATH

    SEQ_LEN = 10 if args.get('seq_len') is None else args['seq_len']
    N_POINTS = 30 if args.get('n_points') is None else args['n_points']
    XY_DIM = 2 if args.get('xy_dim') is None else args['xy_dim']
    PAD_TOKEN_ID = 0 if args.get('pad_token_id') is None else args['pad_token_id']
    VOCAB_SIZE = 17 if args.get('vocab_size') is None else args['vocab_size']

    EMBED_DIM = 192 if args.get('embed_dim') is None else args['embed_dim']
    NUM_HEADS = 12 if args.get('num_heads') is None else args['num_heads']
    NUM_LAYERS = 8 if args.get('num_layers') is None else args['num_layers']
    DIM_FEEDFORWARD = 768 if args.get('dim_feedforward') is None else args['dim_feedforward']
    DROPOUT = 0.1 if args.get('dropout') is None else args['dropout']

    NUM_TIMESTEPS = 1000 if args.get('num_timesteps') is None else args['num_timesteps']
    BETA_START = 0.0001 if args.get('beta_start') is None else args['beta_start']
    BETA_END = 0.02 if args.get('beta_end') is None else args['beta_end']
    SCHEDULE_TYPE = 'cosine' if args.get('schedule_type') is None else args['schedule_type']

    BATCH_SIZE = 128 if args.get('batch_size') is None else args['batch_size']
    VALIDATION_BATCH_SIZE = 2048 if args.get('validation_batch_size') is None else args['validation_batch_size']
    LEARNING_RATE = 4e-5 if args.get('learning_rate') is None else args['learning_rate']
    EPOCHS = 45 if args.get('epochs') is None else args['epochs']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PATIENCE = 10 if args.get('patience') is None else args['patience']
    BEST_MODEL_PATH = "best_d3pm_pointnet_crossattn.pth" if args.get('best_model_path') is None else args['best_model_path']


# ablation study
ablation_tests = [
    {'learning_rate': 1e-4, 'best_model_path': 'ablation_lr_1e-4.pth'},
    {'learning_rate': 5e-5, 'best_model_path': 'ablation_lr_5e-5.pth'},
    {'learning_rate': 1e-5, 'best_model_path': 'ablation_lr_1e-5.pth'},

    {'embed_dim': 128, 'best_model_path': 'ablation_embed_128.pth'},
    {'embed_dim': 192, 'best_model_path': 'ablation_embed_192.pth'},
    {'embed_dim': 256, 'best_model_path': 'ablation_embed_256.pth'},

    {'num_heads': 8, 'best_model_path': 'ablation_heads_8.pth'},
    {'num_heads': 12, 'best_model_path': 'ablation_heads_12.pth'},
    {'num_heads': 16, 'best_model_path': 'ablation_heads_16.pth'},

    {'num_layers': 4, 'best_model_path': 'ablation_layers_4.pth'},
    {'num_layers': 8, 'best_model_path': 'ablation_layers_8.pth'},
    {'num_layers': 12, 'best_model_path': 'ablation_layers_12.pth'},

    {'dim_feedforward': 512, 'best_model_path': 'ablation_ffn_512.pth'},
    {'dim_feedforward': 768, 'best_model_path': 'ablation_ffn_768.pth'},
    {'dim_feedforward': 1024, 'best_model_path': 'ablation_ffn_1024.pth'},

    {'batch_size': 64, 'best_model_path': 'ablation_batch_64.pth'},
    {'batch_size': 128, 'best_model_path': 'ablation_batch_128.pth'},
    {'batch_size': 256, 'best_model_path': 'ablation_batch_256.pth'},
]