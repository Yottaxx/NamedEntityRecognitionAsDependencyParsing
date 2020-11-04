from abc import ABC
from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class SModel(nn.Module, ABC):
    ''' A two-feed-forward-layer module '''

    def __init__(self,vocab_len,d_emb ,d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, d_emb, padding_idx=1)
        self.linear = nn.Linear(d_emb, d_in)
        self.bilstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True, dropout=dropout,
                              bidirectional=bi)
        self.feedStart = feedforwardLayer(d_hid * 2, d_hid * 8,dropout=dropout)
        self.feedEnd = feedforwardLayer(d_hid * 2, d_hid * 8,dropout=dropout)
        self.biaffine = biaffineLayer(d_hid * 2, d_hid * 2, d_class,dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)
        # print(x)
        x = torch.relu(self.linear(x))
        x, _ = self.bilstm(x)
        # print("BILSTM:", x.shape)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        # print("FEEDSTART:", start.shape)
        # print("FEEDEND:", end.shape)
        score = self.biaffine(start, end)
        # print("BIAFFINE:", score.shape)
        return score

#
# if __name__ == "__Main__":
#     x = torch.FloatTensor(3, 4, 10)
#     model = SModel(10, 10, 5,2)
#     result = model(x)
#     print(result.shape)
