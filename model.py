import torch as tc
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMLanguageModel(tc.nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size):
        super(LSTMLanguageModel, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embeddings = tc.nn.Embedding(vocab_size, emb_dim)
        self.lstm = tc.nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = tc.nn.Linear(hidden_dim, vocab_size)

    def initial_state(self, batch_size):
        h = tc.zeros(dtype=tc.float32, size=(1, batch_size, self.hidden_dim)) # one layer lstm; layer idx 1st for state
        c = tc.zeros(dtype=tc.float32, size=(1, batch_size, self.hidden_dim)) # one layer lstm; layer idx 1st for state
        return (h, c)

    def forward(self, padded_sequences, lengths, state=None):
        """Process token sequences using the LSTM, and return sequences of token logits and the final state.

        :param padded_sequences: Tensor of batch-major int sequences, padded to the len of the longest in batch, plus 1.
        :param lengths: Lengths of each unpadded sequence, plus 1.
        :param initial_state: Optional initial state for carrying state across forward calls.
        :return: Tuple containing the token logits in batch-major format, as well as the final state.
        """
        embedded = self.embeddings(padded_sequences)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        batch_size, seq_len = padded_sequences.size()
        initial_state = state if (state is not None) else self.initial_state(batch_size)

        hiddens, final_state = self.lstm(packed_input, initial_state)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True, padding_value=0.0)

        hiddens = hiddens.contiguous()
        hiddens = hiddens.view(-1, self.hidden_dim)
        logits = self.fc(hiddens)

        logprobs = tc.nn.LogSoftmax(dim=-1)(logits)
        logprobs = logprobs.view(batch_size, seq_len, self.vocab_size)
        return logprobs, final_state

    def generate_token(self, prefixes):
        raise NotImplementedError
