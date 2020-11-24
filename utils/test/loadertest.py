from itertools import chain
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import trange


class dataPreLoader:
    def __init__(self, path: str = "./test/", count: int = 3956):
        self.data = self.loadLabel(path, count)

    def loadLabel(self, path: str = "./test/", count: int = 3956) -> list:
        labelList = []
        for i in trange(count):
            sentence = pd.read_csv(path + str(i) + ".txt", names=['Text'])['Text']
            labelList.append(sentence.loc[0])

        return labelList