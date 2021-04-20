import argparse
import torch as tc
import torchtext as tt
from utils import get_tokenizer, get_vocab, text_pipeline, lstm_preprocess_pipeline, ProcessedIterableDataset
from functools import partial
from model import LSTMLanguageModel
from runner import Runner

# Parse arguments.
parser = argparse.ArgumentParser('Pytorch LSTM Language Model')
parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
args = parser.parse_args()

# Datasets.
training_data = tt.datasets.IMDB(root='data', split='train')
test_data = tt.datasets.IMDB(root='data', split='test')

# Preprocessing.
tokenizer = get_tokenizer()
vocab = get_vocab(tokenizer)
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vocab=vocab)
dataset_map = lambda y,x: text_preprocessing(x)
training_data = ProcessedIterableDataset(training_data, dataset_map)
test_data = ProcessedIterableDataset(test_data, dataset_map)

# Dataloaders.
batch_size = 20
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, collate_fn=lstm_preprocess_pipeline)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=lstm_preprocess_pipeline)

# Device.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model.
model = LSTMLanguageModel(emb_dim=128, hidden_dim=128, vocab_size=len(vocab.stoi)).to(device)
print(model)

criterion = tc.nn.NLLLoss()
optimizer = tc.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)

try:
    model.load_state_dict(tc.load("model.pth"))
    optimizer.load_state_dict(tc.load("optimizer.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

# Runner.
runner = Runner(verbose=True)
train_epochs = 10
generate_lines = 10


if args.mode == 'train':
    runner.train(train_epochs, model, train_dataloader, test_dataloader, device, criterion, optimizer)
elif args.mode == 'generate':
    runner.generate(generate_lines)
else:
    raise NotImplementedError


