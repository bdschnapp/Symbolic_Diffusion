import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: reads the input sequence and produces hidden states
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Decoder: attempts to reconstruct the sequence from the encoder's final hidden state
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Output layer to map the decoder's output back to the original coordinate dimensions
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.min_loss = float('inf')
        self.patience = 10
        self.early_stopping_counter = 0

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        # ----- Encoding -----
        # The encoder returns outputs for all timesteps, and the final hidden and cell states
        encoded_seq, (hidden, cell) = self.encoder(x)

        # ----- Preparing Decoder Input -----
        # We create a tensor of zeros as the initial input for the decoder.
        # Alternatively, you could use a special token or the last output of the encoder.
        batch_size, seq_length, _ = x.size()
        decoder_input = torch.zeros(batch_size, seq_length, self.hidden_dim, device=x.device)

        # ----- Decoding -----
        # We initialize the decoder with the encoder's final hidden and cell states
        decoded_seq, _ = self.decoder(decoder_input, (hidden, cell))

        # Map the decoder's output to the original coordinate dimensions
        reconstructed = self.output_layer(decoded_seq)
        return reconstructed

    def train_batch(self, batch):
        if self.early_stopping_counter <= self.patience:
            self.optimizer.zero_grad()
            # Forward pass: The model should output a tensor of the same shape [batch_size, sequence_len, 2]
            output = self(batch[0])

            # Compute the reconstruction loss
            loss = self.criterion(output, batch[0])
            loss.backward()
            self.optimizer.step()

            if loss.item() > self.min_loss:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
            self.min_loss = min(self.min_loss, loss.item())

