import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import defaultdict
from collections import Counter, defaultdict
from transformers import AutoTokenizer

  
class BPE:
    def __init__(self, corpus: list[str], vocab_size: int):
        self.corpus = corpus
        self.vocab_size = vocab_size
        
        #  BERT pre-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
    
    def train(self) -> dict:
        for text in self.corpus:
            pretokenized_words = self._pretokenize(text)
            for word in pretokenized_words:
                self.word_freqs[word] += 1

        all_symbols = sorted({char for word in self.word_freqs for char in word})
        
        vocab = ["</w>"] + all_symbols.copy()
        self.splits = {word: list(word) for word in self.word_freqs.keys()}
        
        while len(vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs()
            most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
            
            self.splits = self.merge_pair(*most_frequent_pair)
            self.merges[most_frequent_pair] = ''.join(most_frequent_pair)
            vocab.append(self.merges[most_frequent_pair])
        
        return self.merges

    def compute_pair_freqs(self) -> defaultdict:
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a: str, b: str) -> dict:
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def tokenize(self, text: str) -> list[str]:
        pretokenized_words = self._pretokenize(text)
        split_words = [list(word) for word in pretokenized_words]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(split_words):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                split_words[idx] = split
        return sum(split_words, [])
    
    def _pretokenize(self, text: str) -> list[str]:
        pretokenized_words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        return [word for word, _ in pretokenized_words_with_offsets]
