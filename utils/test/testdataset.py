from collections import Iterator
from typing import Optional

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoTokenizer, XLNetModel,BertTokenizer
import torch
import numpy as np
from utils.dataStruct import MyDataProcessor
from utils.test.loadertest import dataPreLoader


class TestDataset(Dataset):
    def __init__(self, path: str = "./test/", count: int = 3956):
        self.processor = dataPreLoader(path, count)
        self.data = self.processor.data
        #self.model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
        self.model_path = r'/data/mgliu/transformers_model/roberta_chinese_clue_large'
        self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_clue_large")
        # self.model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=1024)
        self.len = len(self.data)
        #self.max_len = max(list(map(lambda x: len(x.text[0]), self.data)))

    def __getitem__(self, index):

        text = self.data[index]
        if len(text)>510:
            print(len(text), "Cut text into 512 words.")
            text = text[:510]
        raw_text = text
        # label = torch.sparse.FloatTensor(edge, value, torch.Size([len(text)+2, len(text)+2]))
        text = [token for token in text]

        encoding = self.tokenizer.encode_plus(text, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(len(text)+2, input_ids.shape[-1], attention_mask.shape[-1])
        assert (len(text) + 2) == input_ids.shape[-1] and (len(text) + 2) == attention_mask.shape[
            -1], "text tokenizer length matches error."
        # print("input_ids", input_ids.shape)
        # outputs = self.model(input_ids, attention_mask=attention_mask)
        return raw_text, input_ids, attention_mask

    def __len__(self):
        return self.len


class BucketDataLoader(Iterator):
    def __init__(self, dataset: Dataset, batch_size: int = 16, shuffle: bool = True, category: int = 15,
                 train: bool = True):
        self.data = dataset
        self.starts = np.arange(0, len(dataset), batch_size)
        if shuffle:
            np.random.shuffle(self.starts)
        self.num = 0
        self.batch_size = batch_size
        self.max_len = len(dataset)
        self.category = category

    def __next__(self):
        try:
            datalist = range(self.starts[self.num], min(self.max_len, self.batch_size + self.starts[self.num]))
            self.num = self.num + 1
            inputs = []
            attens = []
            for i in datalist:
                ins, atten = self.data[i]
                inputs.append(ins)
                attens.append(atten)
            max_len = max(list(map(lambda x: x.shape[-1], inputs)))
            pad_token = self.data.tokenizer.pad_token_id

            for i in range(len(inputs)):
                inputs[i] = torch.cat((inputs[i],
                                       torch.tensor([pad_token] * (max_len - inputs[i].shape[-1])).unsqueeze(0)), -1)
                attens[i] = torch.cat((attens[i],
                                       torch.tensor([0] * (max_len - attens[i].shape[-1])).unsqueeze(0)), -1)

                # print(inputs[i].shape)

            inputs = torch.stack(inputs, dim=0).squeeze()
            attens = torch.stack(attens, dim=0).squeeze()

            return inputs, attens

        except IndexError:
            self.num = 0
            raise StopIteration()

# dataset = MyDataset(count=2515)
# loader = BucketDataLoader(dataset, 8, True)
# a, b, c = loader.__next__()
