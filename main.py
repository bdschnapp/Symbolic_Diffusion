import train
import eval
from generation import load_model
import config
from data.load_data import load_data, train_val_dataloaders, normalization
from postprocessing.regression import mse_bar_chart

args = {
    'learning_rate': 1e-4,
    'batch_size': 64,
    'embed_dim': 192,
    'dim_feedforward': 512,
    'num_heads': 16,
    'num_layers': 8,
    'epochs': 100,
    'best_model_path': 'hypothetical_optimal_model.pth',

}


def main():
    train_data, val_data = load_data(input_dir='data/Modified', version=7)
    train_loader, val_loader = train_val_dataloaders(train_data, val_data)
    x_mean, x_std, y_mean, y_std = normalization(train_data)

    # Set parameters based on args
    config.set_parameters(args)

    # Train the model
    # train.train(train_loader, val_loader)

    # Load the best model
    loaded_model, diffusion_helper = load_model(args['best_model_path'])

    # Evaluate the model
    mse_list = eval.evaluate_model(
        loaded_model=loaded_model,
        diffusion_helper=diffusion_helper,
        val_data=val_data,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        verbose=True
    )

    # Process and plot the MSE histogram
    mse_bar_chart(mse_list)


if __name__ == "__main__":
    main()
