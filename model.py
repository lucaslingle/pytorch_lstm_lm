import torch as tc
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMLanguageModel(tc.nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size):
        super(LSTMLanguageModel, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embeddings = tc.nn.Embedding(vocab_size, emb_dim)
        self.lstm = tc.nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.fc = tc.nn.Linear(hidden_dim, vocab_size)

    def initial_state(self, batch_size):
        h = tc.zeros(dtype=tc.float32, size=(batch_size, self.hidden_dim))
        c = tc.zeros(dtype=tc.float32, size=(batch_size, self.hidden_dim))
        return (h, c)

    def forward(self, padded_sequences, lengths, initial_state=None):
        """Process token sequences using the LSTM, and return sequences of token logits and the final state.

        :param padded_sequences: Tensor of time-major int sequences, padded to the len of the longest in batch.
        :param lengths: Lengths of each unpadded sequence.
        :param initial_state: Optional initial state for carrying state across forward calls.
        :return: Token logits in time-major format, and the final state.
        """
        embedded = self.embeddings(padded_sequences)
        packed_input = pack_padded_sequence(embedded, lengths)

        batch_size = padded_sequences.shape[1]
        initial_state = initial_state if initial_state is not None else self.initial_state(batch_size)

        hiddens, final_state = self.lstm(packed_input, initial_state)
        logits = self.fc(hiddens)
        return logits, final_state








