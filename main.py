import argparse
import torch as tc
from utils import get_tokenizer, get_vocab, text_pipeline
from functools import partial
from model import LSTMLanguageModel
from runner import Runner

# Parse arguments.
parser = argparse.ArgumentParser('Pytorch LSTM Language Model')
parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
args = parser.parse_args()

# Preprocessing.
tokenizer = get_tokenizer()
vocab = get_vocab(tokenizer)
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vocab=vocab)
dataset_map_fn = lambda y,x: text_preprocessing(x)
batch_size = 20

# Device.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model.
model = LSTMLanguageModel(emb_dim=128, hidden_dim=128, vocab_size=len(vocab.stoi)).to(device)
optimizer = tc.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)

try:
    model.load_state_dict(tc.load("model.pth"))
    optimizer.load_state_dict(tc.load("optimizer.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

# Runner.
runner = Runner(verbose=True)
epochs = 10

if args.mode == 'train':
    runner.train(dataset_map_fn, batch_size, epochs, model, device, optimizer)
elif args.mode == 'generate':
    runner.generate()
else:
    raise NotImplementedError


