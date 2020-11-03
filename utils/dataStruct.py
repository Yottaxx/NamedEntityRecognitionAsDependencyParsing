from utils.Loader import dataPreLoader


def process(line):
    return line['Category'].to_list()


class DataStruct:

    def __init__(self, data):
        self.edgeLeft = data['Pos_b'].tolist()
        self.edgeRight = data['Pos_e'].tolist()
        self.values = data['Category'].tolist()
        self.text = [data['Text'].tolist()[0]]


class MyDataLoader:
    def __init__(self,path: str = "./train/", count: int = 10):
        self.data = list(map(lambda line: DataStruct(line), dataPreLoader(path,count).data))

    def loadData(self):
        return self.data


