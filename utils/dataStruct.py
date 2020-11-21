from utils.Loader import dataPreLoader


def process(line):
    return line['Category'].to_list()


class DataStruct:

    def __init__(self, data):
        self.edgeLeft = data['Pos_b'].tolist()
        self.edgeRight = data['Pos_e'].tolist()
        self.values = data['Category'].tolist()
        self.text = [data['Text'].tolist()[0]]
        #self.label_num = len(data)


class MyDataProcessor:
    def __init__(self, path: str = "./train/", count: int = 10):
        self.preLoader = dataPreLoader(path, count)
        self.cateDict = self.preLoader.index2cate
        #print("mydataProcessor")
        print(self.preLoader.index2cate)
        self.data = list(map(lambda line: DataStruct(line), self.preLoader.data))

    def loadData(self):
        return self.data

    def loadDict(self):
        return self.cateDict
