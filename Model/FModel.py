from abc import ABC

from transformers import XLNetModel, BertModel

from Layers import feedforwardLayer, biaffineLayer
import torch.nn as nn
import torch.nn.functional as F
import torch


class FModel(nn.Module, ABC):

    def __init__(self, d_in, d_hid, d_class, n_layers, bi=True, dropout=0.3):
        super().__init__()
        # self.model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
        # self.model_path = r'/data/mgliu/transformers_model/chinese-xlnet-mid'
        # self.model_path = r'/data/mgliu/transformers_model/roberta_chinese_clue_large'
        self.model = BertModel.from_pretrained("clue/roberta_chinese_clue_large")
        # self.model = BertModel.from_pretrained(self.model_path)
        # self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)
        self.bilstm = nn.LSTM(d_in, 200, num_layers=3, batch_first=True, dropout=0.4,
                              bidirectional=bi)
        self.feedStart = feedforwardLayer(400, 150, dropout=0.2)
        self.feedEnd = feedforwardLayer(400, 150, dropout=0.2)
        self.biaffine = biaffineLayer(400, 400, d_class, dropout=dropout)

    def forward(self, x, atten):
        x = self.model(x, atten)[0]
        # print(x.shape)
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
