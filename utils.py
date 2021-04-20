import torch as tc
import torchtext as tt
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def get_tokenizer():
    tokenizer = tt.data.utils.get_tokenizer('basic_english')
    return tokenizer

def get_vocab(tokenizer):
    # https://pytorch.org/text/stable/vocab.html
    # vocab object will have stoi, a defaultdict with ('<unk>', '<pad>', other tokens) mapping to (0, 1, ...).
    # lookups for unknown tokens will default to the '<unk>' key, and thus the token 0 as the value.
    train_iter = tt.datasets.IMDB(root='data', split='train')
    counter = Counter()
    for y, X in train_iter:
        counter.update(tokenizer(X))
    vocab = tt.vocab.Vocab(counter, specials=('<unk>', '<pad>', '<go>'), min_freq=50)
    return vocab

def text_pipeline(text, tokenizer, vocab):
    sequence = [vocab.stoi[token] for token in tokenizer(text)]
    return sequence

def lstm_preprocess_pipeline(sequences, max_tokens=20):
    batch_size = len(sequences)
    sequences = [tc.Tensor(s)[0:max_tokens] for s in sequences]

    padded = pad_sequence(sequences, padding_value=1.0, batch_first=True) # <pad> is token at index 1 of vocab.
    go_tokens = 2 * tc.ones(size=(batch_size, 1)) # <go> is token at index 2.
    padded = tc.cat((go_tokens, padded), dim=1) # prepend <go> tokens.
    pad_tokens = tc.ones(size=(batch_size, 1)) # <pad> is token at index 1.
    padded = tc.cat((padded, pad_tokens), dim=1) # append one extra <pad> token to each seq. to ensure eos targ.
    padded = padded.long()

    input_tokens = padded[:, 0:-1]
    target_tokens = padded[:, 1:]
    lengths = tc.Tensor([len(s)+1 for s in sequences]) # add 1 to ensure num lstm iters includes input length w go token

    return input_tokens, target_tokens, lengths


class ProcessedIterableDataset(tc.utils.data.IterableDataset):
    # This is cleaner than using collate_fn in the dataloader.
    # Investigation shows it is faster, as well.
    # It also lets you shuffle the data, which the dataloader itself does not support for IterableDatasets.
    def __init__(self, dataset, function):
        self.dataset = tc.utils.data.BufferedShuffleDataset(dataset, buffer_size=25000)
        self.function = function

    def __iter__(self):
        return (self.function(x,y) for x,y in self.dataset.__iter__())