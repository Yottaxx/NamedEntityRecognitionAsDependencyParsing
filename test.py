from utils.Loader import dataPreLoader

path = './utils/train/'
count = 2515
preLoader = dataPreLoader(path, count)

data = preLoader.data

print(len(data))

print(data[0])