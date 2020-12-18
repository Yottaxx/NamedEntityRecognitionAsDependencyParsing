from itertools import chain
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import trange


class dataPreLoader:
    def __init__(self, path: str = "./train/", count: int = 2515):
        self.data, self.index2cate, self.cate2index = self.dataFilter(path, count)

    def loadLabel(self, path: str = "./train/", count: int = 2515) -> list:
        labelList = []
        for i in trange(count):
            # 跳过两个会报错的label文件
            if i == 30 or i == 667:
                continue
            temp = pd.read_csv(path + "label/" + str(i) + ".csv")
            sentence = pd.read_csv(path + "data/" + str(i) + ".txt", names=['Text'])['Text']
            sentence = ''.join(list(sentence))
            temp['Text'] = sentence[:min(len(sentence), 500)]
            temp = temp[temp["Pos_b"] < 500][temp["Pos_e"] < 500]
            labelList.append(temp)
            # print(temp)
        return labelList

    # def tokenizer(sequence: str) -> list:
    #     return [tok for tok in sequence]

    # def loadData(self,path: str = "./train/data/", count: int = 2515) -> list:
    #     dataList = []
    #     for i in trange(count):
    #         temp = pd.read_csv(path + str(i) + ".txt", names=['Text'])
    #         dataList.append(temp.loc[0]['Text'])
    #         # print(temp.loc[0]['text'])
    #     # dataList = list(map(lambda text: tokenizer(text), dataList))
    #     return dataList

    def match(self, data: str, label: pd) -> bool:
        if data[int(label["Pos_b"]):int(label["Pos_e"]) + 1] != str(label["Privacy"]):
            print(data[int(label["Pos_b"]):int(label["Pos_e"]) + 1], str(label["Privacy"]))
            print(len(data[int(label["Pos_b"]):int(label["Pos_e"]) + 1]), len(str(label["Privacy"])))
            return False
        return True
        # assert data[int(label["Pos_b"]):int(label["Pos_e"]) + 1] == str(label["Privacy"]), "data matches"

    # 判断是否开始位置结束位置实体 为标注的单词段落
    def matches(self, label: pd) -> pd:
        # print(label)
        if label["Text"][int(label["Pos_b"]):int(label["Pos_e"]) + 1] != str(label["Privacy"]):
            print(label["Text"][int(label["Pos_b"]):int(label["Pos_e"]) + 1], str(label["Privacy"]))
            print(len(label["Text"][int(label["Pos_b"]):int(label["Pos_e"]) + 1]), len(str(label["Privacy"])))
            return None
        return label

    def funcMatch(self, label: pd) -> pd:
        label = label.apply(lambda x: self.matches(x), axis=1).dropna(axis=0, how='any')
        return label

    def load(self, path: str = "./train/", count: int = 2515) -> list:
        data = self.loadLabel(path, count)
        data = list(map(lambda atom: self.funcMatch(atom), data))
        return data

    def dataFilter(self, path: str = "./train/", count: int = 2515) -> Tuple[list, Dict[int, Any], Dict[Any, int]]:
        def process(line):
            if len(line) >= 1:
                return line['Category'].to_list()
            return []

        def filterCategory(line: pd):
            if len(line) >= 1:
                line = line[line['Category'].isin(category)]
                return line
            return None

        data = self.load(path, count)

        # category = list(map(lambda line: process(line), data))
        # category = list(sorted(set(list(chain(*category)))))
        category = ['QQ', 'address', 'book', 'company', 'email',
                    'game', 'government', 'mobile', 'movie',
                    'name', 'organization', 'position', 'scene', 'vx']

        data = list(map(lambda line: filterCategory(line), data))

        index = [i for i in range(len(category))]
        cate2index = dict(zip(category, index))
        index2cate = dict(zip(index, category))
        oneHot = np.eye(len(index)).tolist()
        print(category)

        dataC = []
        for d in data:
            if d is not None:
                d = d.dropna(axis=0, how='any')
                d = d.reset_index()
                dataC.append(d)
        data = dataC

        for d in data:
            for i in range(len(d)):
                assert d.loc[i]['Category'] in category

        def cate2oneHotProcessing(line):
            # print(line)
            if line['Category'] in category:
                line['Category'] = cate2index[line["Category"]] + 1
            # return self.oneHot(cate2index[line["Category"]])
            return line

        def cate2oneHot(line):
            line = line.apply(lambda x: cate2oneHotProcessing(x), axis=1)
            return line

        data = list(map(lambda line: cate2oneHot(line), data))
        print("total_data_length:", len(data))
        return data, index2cate, cate2index
