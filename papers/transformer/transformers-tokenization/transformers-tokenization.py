class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
            
        # Exclude special tokens if they happen to appear in the raw text
        sorted_words = sorted(unique_words - set(special_tokens))
        
        all_tokens = special_tokens + sorted_words
        self.word_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_word = {idx: token for idx, token in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join(self.id_to_word.get(idx, self.unk_token) for idx in ids)
