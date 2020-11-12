from collections import Iterator
from typing import Optional

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoTokenizer, XLNetModel
import torch
import numpy as np
from utils.dataStruct import MyDataProcessor


class MyDataset(Dataset):
    def __init__(self, path: str = "./train/", count: int = 2515):
        self.processor = MyDataProcessor(path, count)
        self.data = self.processor.loadData()
        self.cateDict = self.processor.loadDict()
        #self.model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
        self.model_path = r'/data/lingvo_data/transformers_model/chinese-xlnet-mid'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        #self.model = XLNetModel.from_pretrained(self.model_path, mem_len=768)
        self.len = len(self.data)
        self.max_len = max(list(map(lambda x: len(x.text[0]), self.data)))

    def __getitem__(self, index):

        text = self.data[index].text[0]
        value = torch.tensor(self.data[index].values)

        edgeLeft = torch.tensor(self.data[index].edgeLeft)
        edgeRight = torch.tensor(self.data[index].edgeRight)

        if len(value.shape) == 1:
            value.unsqueeze(0)
        if len(edgeLeft.shape) == 1:
            edgeRight.unsqueeze(0)
        if len(edgeLeft.shape) == 1:
            edgeRight.unsqueeze(0)

        edge = torch.stack([edgeLeft, edgeRight], dim=0).long()
        # label = torch.sparse.FloatTensor(edge, value, torch.Size([len(text)+2, len(text)+2]))
        text = [token for token in text]

        encoding = self.tokenizer.encode_plus(text, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(len(text)+2, input_ids.shape[-1], attention_mask.shape[
        #     -1])
        assert (len(text) + 2) == input_ids.shape[-1] and (len(text) + 2) == attention_mask.shape[
            -1], "text tokenizer length matches error."
        # print("input_ids", input_ids.shape)
        # outputs = self.model(input_ids, attention_mask=attention_mask)
        return input_ids, attention_mask, edge, value

    def __len__(self):
        return self.len


class BucketDataLoader(Iterator):
    def __init__(self, dataset: Dataset, batch_size: int = 16, shuffle: bool = True, category: int = 15,
                 train: bool = True):
        self.data = dataset
        self.starts = np.arange(0, len(dataset), batch_size)
        if shuffle:
            np.random.shuffle(self.starts)
        if train:
            self.starts = self.starts[:int(3 * len(self.starts) / 4)]
        else:
            self.starts = self.starts[int(3 * len(self.starts) / 4):]

        self.num = 0
        self.batch_size = batch_size
        self.max_len = len(dataset)
        self.category = category

    def __next__(self):
        try:
            datalist = range(self.starts[self.num], min(self.max_len, self.batch_size + self.starts[self.num]))
            self.num = self.num + 1
            inputs = []
            edges = []
            values = []
            label = []
            attens = []
            for i in datalist:
                ins, atten, edge, value = self.data[i]
                inputs.append(ins)
                attens.append(atten)
                edges.append(edge)
                values.append(value)
            max_len = max(list(map(lambda x: x.shape[-1], inputs)))
            pad_token = self.data.tokenizer.pad_token_id

            for i in range(len(inputs)):
                inputs[i] = torch.cat((inputs[i],
                                       torch.tensor([pad_token] * (max_len - inputs[i].shape[-1])).unsqueeze(0)), -1)
                attens[i] = torch.cat((attens[i],
                                       torch.tensor([0] * (max_len - attens[i].shape[-1])).unsqueeze(0)), -1)
                label.append(torch.sparse.FloatTensor(edges[i], values[i], torch.Size([max_len, max_len])).to_dense())
                # oneHot = torch.zeros((label[i].shape[0], label[i].shape[1], self.category))
                # oneHot.scatter_(2, label[i].to_dense().long().unsqueeze(-1), 1)
                # label[i] = oneHot
                # print(inputs[i].shape)

            inputs = torch.stack(inputs, dim=0).squeeze()
            attens = torch.stack(attens, dim=0).squeeze()
            label = torch.stack(label, dim=0)

            return inputs, attens, label

        except IndexError:
            np.random.shuffle(self.starts)
            self.num = 0
            raise StopIteration()

# dataset = MyDataset(count=2515)
# loader = BucketDataLoader(dataset, 8, True)
# a, b, c = loader.__next__()
