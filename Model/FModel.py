from abc import ABC

from transformers import XLNetModel

from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class FModel(nn.Module, ABC):

    def __init__(self, d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()
        # self.model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
        self.model_path = r'/data/mgliu/transformers_model/chinese-xlnet-mid'
        self.model = XLNetModel.from_pretrained(self.model_path, mem_len=768)
        # self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)

        # self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)
        # self.bilstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True, dropout=dropout,
        #                       bidirectional=bi)
        self.feed = nn.Linear(d_in, d_hid)
        self.transformer = nn.Transformer(d_model=d_hid, nhead=5, num_encoder_layers=2, num_decoder_layers=2,
                                          dropout=dropout, dim_feedforward=d_hid * 2)
        self.feedStart = feedforwardLayer(d_hid, d_hid * 2, dropout=0.2)
        self.feedEnd = feedforwardLayer(d_hid, d_hid * 2, dropout=0.2)
        self.biaffine = biaffineLayer(d_hid, d_hid, d_class, dropout=dropout)

    def forward(self, x, atten):
        x = self.model(x, atten)[0]
        x = self.feed(x)
        x = self.transformer(src=x.transpose(0, 1), tgt=x.transpose(0, 1), src_key_padding_mask=(atten.bool()),
                             tgt_key_padding_mask=(atten.bool()))
        x = x.transpose(0, 1)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        score = self.biaffine(start, end)
        return score

#
# if __name__ == "__Main__":
#     x = torch.FloatTensor(3, 4, 10)
#     model = SModel(10, 10, 5,2)
#     result = model(x)
#     print(result.shape)
