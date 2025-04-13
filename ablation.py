import config
from generation import load_model
from data.load_data import load_data, normalization, train_val_dataloaders

ablation_tests = [
    {'learning_rate': 1e-4, 'best_model_path': 'ablation/lr_1e-4.pth'},
    {'learning_rate': 5e-5, 'best_model_path': 'ablation/lr_5e-5.pth'},
    {'learning_rate': 1e-5, 'best_model_path': 'ablation/lr_1e-5.pth'},

    {'embed_dim': 132, 'best_model_path': 'ablation/embed_132.pth'},
    {'embed_dim': 192, 'best_model_path': 'ablation/embed_192.pth'},
    {'embed_dim': 264, 'best_model_path': 'ablation/embed_264.pth'},

    {'num_heads': 8, 'best_model_path': 'ablation/heads_8.pth'},
    {'num_heads': 12, 'best_model_path': 'ablation/heads_12.pth'},
    {'num_heads': 16, 'best_model_path': 'ablation/heads_16.pth'},

    {'num_layers': 4, 'best_model_path': 'ablation/layers_4.pth'},
    {'num_layers': 8, 'best_model_path': 'ablation/layers_8.pth'},
    {'num_layers': 12, 'best_model_path': 'ablation/layers_12.pth'},

    {'dim_feedforward': 512, 'best_model_path': 'ablation/ffn_512.pth'},
    {'dim_feedforward': 768, 'best_model_path': 'ablation/ffn_768.pth'},
    {'dim_feedforward': 1024, 'best_model_path': 'ablation/ffn_1024.pth'},

    {'batch_size': 64, 'best_model_path': 'ablation/batch_64.pth'},
    {'batch_size': 128, 'best_model_path': 'ablation/batch_128.pth'},
    {'batch_size': 256, 'best_model_path': 'ablation/batch_256.pth'},
]


def train_ablation():
    import train
    train_data, val_data = load_data(input_dir='data/Modified', version=7)
    train_loader, val_loader = train_val_dataloaders(train_data, val_data)

    for test in ablation_tests:
        config.set_parameters(test)
        train.train(train_loader, val_loader)


def eval_ablation():
    import eval
    train_data, val_data = load_data(input_dir='data/Modified', version=7)
    x_mean, x_std, y_mean, y_std = normalization(train_data)

    for test in ablation_tests:
        config.set_parameters(test)
        loaded_model, diffusion_helper = load_model(test['best_model_path'])

        mse = eval.evaluate_model(
            loaded_model=loaded_model,
            diffusion_helper=diffusion_helper,
            val_data=val_data,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            verbose=False,
        )

        print(f"Model: {test['best_model_path']}, MSE: {mse}")


if __name__ == "__main__":
    train_ablation()
    eval_ablation()
