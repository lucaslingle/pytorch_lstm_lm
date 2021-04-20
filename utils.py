import torch as tc
import torchtext as tt
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def get_tokenizer():
    tokenizer = tt.data.utils.get_tokenizer('basic_english')
    return tokenizer

def get_vocab(train_iter, tokenizer):
    # https://pytorch.org/text/stable/vocab.html
    # vocab object will have stoi, a defaultdict with ('<unk>', '<pad>', other tokens) mapping to (0, 1, ...).
    # lookups for unknown tokens will default to the '<unk>' key, and thus the token 0 as the value.
    counter = Counter()
    for y, X in train_iter:
        counter.update(tokenizer(X))
    vocab = tt.vocab.Vocab(counter)
    return vocab

def text_pipeline(text, tokenizer, vocab):
    sequence = [vocab.stoi[token] for token in tokenizer(text)]
    return sequence

def lstm_preprocess_pipeline(sequences):
    padded = pad_sequence(sequences, padding_value=1) # <pad> is token at index 1.
    batch_size = len(sequences)
    shifted = tc.cat((padded[:, 1:], tc.ones(size=(batch_size,1))), dim=-1)
    timemajor_inputs = padded.transpose(1, 0)
    timemajor_targets = shifted.transpose(1, 0)
    return timemajor_inputs, timemajor_targets

class ProcessedIterableDataset(tc.utils.data.IterableDataset):
    # This is cleaner than using collate_fn in the dataloader.
    # Investigation shows it is faster, as well.
    # It also lets you shuffle the data, which the dataloader itself does not support for IterableDatasets.
    def __init__(self, dataset, function):
        self.dataset = tc.utils.data.BufferedShuffleDataset(dataset, buffer_size=25000)
        self.function = function

    def __iter__(self):
        return (self.function(x,y) for x,y in self.dataset.__iter__())