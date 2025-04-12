from generation import generate_tokens, load_model
from data.load_data import load_data, normalization
from postprocessing.tokenizer import MathTokenizer
from postprocessing.regression import process_and_plot
import numpy as np
import config


def main():
    # Load the data
    train_data, val_data = load_data(input_dir='data/Modified', version=7)

    # Normalize the data
    x_mean, x_std, y_mean, y_std = normalization(train_data)

    # load the model
    loaded_model, diffusion_helper = load_model("best_d3pm_pointnet_crossattn.pth")

    # Generate data using the trained model
    for item_index_to_generate in range(24, 54):

        data_item_to_generate = val_data[item_index_to_generate]

        # 4. Call the generation function
        generated_tokens = generate_tokens(
            data_item=data_item_to_generate,
            model=loaded_model,
            diffusion_helper=diffusion_helper,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=config.DEVICE,
            seq_len=config.SEQ_LEN,
            n_points=config.N_POINTS,
            xy_dim=config.XY_DIM
        )

        true_tokens = np.array(data_item_to_generate['token_ids'])
        print(f"Ground Truth Tokens:\n{true_tokens}")

        print(f"Generated tokens: {generated_tokens}")

        tokenizer = MathTokenizer()

        # Discard all tokens after the first occurrence of 2
        if 2 in generated_tokens:
            generated_tokens = generated_tokens[:np.where(generated_tokens == 2)[0][0] + 1]

        # Decode the generated tokens
        decoded_expression = tokenizer.decode(generated_tokens)
        decoded_ground_truth = tokenizer.decode(true_tokens)
        print(f"Decoded Ground Truth Expression: {decoded_ground_truth}")

        print(f"Decoded Expression: {decoded_expression}")

        # Split the 30x2 array into two 30x1 arrays and save as keys "X" and "Y"
        data_item_to_generate["X"] = [coord[0] * x_std + x_mean for coord in data_item_to_generate["X_Y_combined"]]
        data_item_to_generate["Y"] = [coord[1] * y_std + y_mean for coord in data_item_to_generate["X_Y_combined"]]

        # Write the decoded expression to the "RPN" key
        data_item_to_generate["RPN"] = decoded_expression

        process_and_plot(data_item_to_generate)


if __name__ == "__main__":
    main()
