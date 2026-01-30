import torch
import math
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hid_dim)

        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()

        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(
                self.emb_dim, self.hid_dim
            ).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(
                self.hid_dim, self.hid_dim
            ).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return h, c

    def detach_hidden(self, hidden):
        h, c = hidden
        return h.detach(), c.detach()

    def forward(self, src, hidden):
        emb = self.dropout(self.embedding(src))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden