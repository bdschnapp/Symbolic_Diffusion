class MathTokenizer:
    def __init__(self, vocab=None):
        """
        If no vocabulary is provided, build a basic one including special tokens.
        You can later update the vocabulary by processing your entire dataset.
        """
        if vocab is None:
            # Special tokens: PAD, SOS (start of sequence), EOS (end of sequence), and UNK (unknown)
            self.vocab = {
                "<PAD>": 0,
                "<SOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3,
            }
            # You can pre-populate the vocabulary with common tokens that appear in your RPN equations.
            for token in ["C", "x1", "sin", "cos", "exp", "log", "+", "-", "*", "/","**","2","3"]:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        else:
            self.vocab = vocab
        # Create an inverse vocabulary for decoding.
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, expression):
        """
        Splits the RPN expression into tokens.
        Assumes tokens are separated by whitespace.
        For example, "C C x1 * C + sin * C +" becomes:
          ["C", "C", "x1", "*", "C", "+", "sin", "*", "C", "+"]
        """
        tokens = expression.strip().split()
        return tokens

    def build_vocab_from_list(self, expressions):
        """
        Given a list of RPN equation strings, update the vocabulary
        to include any token that is not already present.
        """
        for expr in expressions:
            tokens = self.tokenize(expr)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        # Rebuild inverse vocabulary.
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, expression, add_special_tokens=True):
        """
        Tokenize and encode an RPN expression into a list of token IDs.
        Optionally add <SOS> at the beginning and <EOS> at the end.
        """
        tokens = self.tokenize(expression)
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab["<SOS>"])
        for token in tokens:
            # Use <UNK> if token not in vocab.
            token_ids.append(self.vocab.get(token, self.vocab["<UNK>"]))
        if add_special_tokens:
            token_ids.append(self.vocab["<EOS>"])
        return token_ids

    def decode(self, token_ids, remove_special_tokens=True):
        """
        Convert a list of token IDs back into a token sequence (string).
        Optionally remove special tokens (<SOS>, <EOS>, and <PAD>).
        """
        tokens = []
        for tid in token_ids:
            token = self.inv_vocab.get(tid, "<UNK>")
            if remove_special_tokens and token in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            tokens.append(token)
        return " ".join(tokens)