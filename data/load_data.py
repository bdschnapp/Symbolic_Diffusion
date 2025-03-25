import os
import json
import psutil
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import datasets
from datasets import Dataset as Dataset2
from data.preprocess import infix_to_rpn, process_and_plot

"""
def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):

    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            sampler=sampler,
            # shuffle=not deterministic,
            num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            # sampler=sampler,
            shuffle=not deterministic,
            num_workers=4,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)
"""


def load_data(
        batch_size,
        seq_len,
        deterministic=False,
        data_args=None,
        model_emb=None,
        split='train',
        loaded_vocab=None,
        loop=True,
        verbose=False
):
    """
    Load data and prepare it for the diffusion model training process.
    Returns an iterator over batches of data.
    If a cached tokenized dataset exists, it loads it from disk;
    otherwise, it processes and tokenizes the data normally and then saves it.
    """
    print('#' * 30, f'\nLoading {split} data...')

    # Determine the path for the cached tokenized dataset.
    # Here we assume that data_args has a "checkpoint_path" attribute.
    saved_data_file = f"data/tokenized_data_{split}.pkl"

    if os.path.exists(saved_data_file):
        print("Found cached dataset at:", saved_data_file)
        with open(saved_data_file, "rb") as f:
            tokenized_data = pickle.load(f)
    else:
        print("No cached dataset found. Loading raw data...")
        # Load raw data from CSV file
        data = load_data_csv(split)

        # Process each record: convert infix notation to RPN if needed.
        for record in data:
            if "Skeleton" in record and "RPN" not in record:
                skeleton_expr = record["Skeleton"]
                rpn_expr = infix_to_rpn(skeleton_expr)
                record["RPN"] = rpn_expr

        # Create dataset dictionary in the format expected by tokenize_helper.
        dataset_dict = {
            'src_x': [record["X"] for record in data],
            'src_y': [record["Y"] for record in data],
            'trg': [record["RPN"] for record in data]
        }

        if verbose:
            print(f"Total records: {len(data)}")
            print(f"Sample RPN: {data[0]['RPN']}")
            print(f"Sample X length: {len(data[0]['X'])}")
            print(f"Sample Y length: {len(data[0]['Y'])}")
            print(f"Vocabulary size: {loaded_vocab.vocab_size}")

        # Process and tokenize the dataset
        tokenized_data = tokenize_helper(dataset_dict, loaded_vocab, seq_len)

        # Save the tokenized data to file on rank 0 only.
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print("Saving tokenized data to:", saved_data_file)
            # Ensure the checkpoint path exists.
            os.makedirs(data_args.checkpoint_path, exist_ok=True)
            with open(saved_data_file, "wb") as f:
                pickle.dump(tokenized_data, f)

    # Create dataset object and dataloader
    dataset = CustomDataset(
        tokenized_data,
        data_args,
        model_emb=model_emb
    )

    if split != 'test':
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not deterministic,
            num_workers=4,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        return iter(data_loader)


def tokenize_helper(sentence_lst, vocab_dict, seq_len):
    """
    Convert the dataset dictionary into a tokenized form ready for model training.
    Directly use X, Y coordinate float arrays as source inputs and tokenized RPN strings as targets.
    """
    # Monitor memory usage
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Create dataset object from dictionary
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)

    # Define tokenization function
    def tokenize_function(examples):
        # We'll only tokenize the RPN target strings
        # The X,Y coordinates will be handled separately

        input_id_y = []

        # Process each example
        for i in range(len(examples['trg'])):
            # For target: encode the RPN string using the tokenizer
            # This applies the RPN tokenization as seen in Database_Test.ipynb
            id_y = vocab_dict.encode(examples['trg'][i], add_special_tokens=True)
            input_id_y.append(id_y)

        return {'input_id_y': input_id_y}

    # Apply tokenization to dataset
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print('### tokenized_datasets', tokenized_datasets)

    # Function to prepare X,Y coordinates and create masks
    def prepare_coords_and_mask(group_lst):
        lst = []
        mask = []
        coords = []

        for i in range(len(group_lst['input_id_y'])):
            # Extract target sequence
            trg = group_lst['input_id_y'][i]

            # Get the x,y coordinates
            x_values = group_lst['src_x'][i]
            y_values = group_lst['src_y'][i]

            # Combine x,y into a single array of coordinate pairs
            coord_pairs = []
            for j in range(min(len(x_values), len(y_values))):
                # Get individual values, handling both scalar and list cases
                x = x_values[j] if not isinstance(x_values[j], list) else x_values[j][0]
                y = y_values[j] if not isinstance(y_values[j], list) else y_values[j][0]

                # Ensure they're float values
                try:
                    x_float = float(x)
                    y_float = float(y)
                    coord_pairs.append([x_float, y_float])
                except (ValueError, TypeError):
                    # Skip invalid coordinates
                    print(f"Skipping invalid coordinate: x={x}, y={y}")

            # Ensure we don't exceed maximum sequence length
            max_coord_pairs = (seq_len - len(trg) - 1) // 2
            if len(coord_pairs) > max_coord_pairs:
                coord_pairs = coord_pairs[:max_coord_pairs]

            # Store the coordinate pairs
            coords.append(coord_pairs)

            # Create the target token sequence for this example
            lst.append(trg)

            # Create mask (0 for target tokens to predict)
            mask.append([0] * len(trg))

        group_lst['coords'] = coords  # Raw coordinate pairs
        group_lst['input_ids'] = lst  # Target token IDs
        group_lst['input_mask'] = mask  # Mask for target tokens
        return group_lst

    # Apply coordinate preparation and mask creation
    tokenized_datasets = tokenized_datasets.map(
        prepare_coords_and_mask,
        batched=True,
        num_proc=1,
        desc="preparing coordinates and masks",
    )

    # Define padding function
    def pad_function(group_lst):
        max_length = seq_len

        # Pad the target token IDs
        group_lst['input_ids'] = _collate_batch_helper(
            group_lst['input_ids'],
            vocab_dict.vocab["[PAD]"],
            max_length
        )

        # Pad the masks
        group_lst['input_mask'] = _collate_batch_helper(
            group_lst['input_mask'],
            1,  # Mask value for padded positions
            max_length
        )

        # For coordinates, we need a different approach since they're 2D arrays
        # We'll standardize the number of coordinate pairs and pad with zeros
        max_coords = max(len(coords) for coords in group_lst['coords'])
        padded_coords = []

        for coord_list in group_lst['coords']:
            # Pad with zero coordinates if needed
            if len(coord_list) < max_coords:
                padded = coord_list + [[0.0, 0.0]] * (max_coords - len(coord_list))
            else:
                padded = coord_list
            padded_coords.append(padded)

        group_lst['coords'] = padded_coords
        return group_lst

    # Apply padding to dataset
    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc="padding",
    )

    print(lm_datasets, 'padded dataset')

    # Create final dataset dictionary
    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets

    return raw_datasets


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def infinite_loader(data_loader):
    """Create an infinite iterator over a DataLoader"""
    while True:
        yield from data_loader


def load_data_csv(split):
    # Load dataset json
    data = []
    f = f"data/{split}.json"
    with open(f, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Now data is a list of dictionaries, each containing one record
    print(f"Loaded {len(data)} records")

    # Remove entries that contain the sequence "Ce" in the "Skeleton" key
    data = [record for record in data if "Ce" not in record["Skeleton"]]

    # Print the number of records after removal
    print(f"Number of records after removal: {len(data)}")

    return data


class CustomDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            # Get the target token IDs and convert to tensor
            input_ids = self.text_datasets['train'][idx]['input_ids']

            # Get the raw coordinate data
            coords = self.text_datasets['train'][idx]['coords']

            # Convert coordinates to tensor (will be shape [n_points, 2])
            coords_tensor = torch.tensor(coords, dtype=torch.float32)

            # Apply any required transformations or normalization to coordinates
            # Here I'm assuming they're already normalized, but you might need to adjust

            # If your model expects a different format, convert here
            # For example, flatten the coordinates to [n_points * 2]
            # coords_flat = coords_tensor.reshape(-1)

            # The diffusion model will take coords_tensor as input
            # The target is the tokenized RPN expression (input_ids)

            # Prepare output dictionary
            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(input_ids)
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

            # Return the coordinates as the primary input to the diffusion model
            # and the token-related data as kwargs
            return coords_tensor, out_kwargs
