from abc import ABC

from transformers import XLNetModel

from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class FModel(nn.Module, ABC):

    def __init__(self, d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()

        with torch.no_grad():
            self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)

        self.bilstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True, dropout=dropout,
                              bidirectional=bi)
        self.feedStart = feedforwardLayer(d_hid * 2, d_hid * 8, dropout=0.2)
        self.feedEnd = feedforwardLayer(d_hid * 2, d_hid * 8, dropout=0.2)
        self.biaffine = biaffineLayer(d_hid * 2, d_hid * 2, d_class, dropout=dropout)

    def forward(self, x, atten):
        x = self.model(x, atten)[0]
        print(x.shape)
        x, _ = self.bilstm(x)
        # print("BILSTM:", x)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        # print("FEEDSTART:", start)
        # print("FEEDEND:", end)
        score = self.biaffine(start, end)
        # print("BIAFFINE:", score)
        return score

#
# if __name__ == "__Main__":
#     x = torch.FloatTensor(3, 4, 10)
#     model = SModel(10, 10, 5,2)
#     result = model(x)
#     print(result.shape)
