import json


class MathTokenizer:
    def __init__(self, args):
        vocab_path = args.vocab

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Create an inverse vocabulary for decoding.
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.vocab_size = len(self.vocab)

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

        self.vocab_size = len(self.vocab)

    """
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
    """

    def encode(self, expression, add_special_tokens=True):
        tokens = self.tokenize(expression)
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab["[SOS]"])
        for token in tokens:
            # Use <UNK> if token not in vocab.
            token_ids.append(self.vocab.get(token, self.vocab["[UNK]"]))
        if add_special_tokens:
            token_ids.append(self.vocab["[EOS]"])
        return token_ids

    def encode_token(self, expression):
        """
        Tokenize and encode an RPN expression into a list of token IDs.
        Optionally add [SOS] at the beginning and [EOS] at the end.
        """
        if isinstance(expression, str):
            return self.encode(expression)
        elif isinstance(expression, list):
            return [self.encode(expr) for expr in expression]

    """
        def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens
    """

    def decode(self, token_ids):
        """
        Convert a list of token IDs back into a token sequence (string).
        Optionally remove special tokens ([SOS], [EOS], and [PAD]).
        """
        remove_special_tokens = True
        tokens = []
        for tid in token_ids:
            token = self.inv_vocab.get(tid, "[UNK]")
            if remove_special_tokens and token in ["[SOS]", "[EOS]", "[PAD]"]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def decode_token(self, token_ids):
        if isinstance(token_ids, list) and all(isinstance(i, int) for i in token_ids):
            return [self.decode(ids) for ids in token_ids]
        else:
            return self.decode(token_ids)


def load_tokenizer(args):
    # TODO:
    tokenizer = MathTokenizer(args)
    return tokenizer
