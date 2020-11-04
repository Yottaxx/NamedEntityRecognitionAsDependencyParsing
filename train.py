from tqdm import trange

from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn

epoch = 8
batch_size = 8
dataset = MyDataset(path="./utils/train/", count=2515)
trainLoader = BucketDataLoader(dataset, batch_size, True,True)
devLoader = BucketDataLoader(dataset, batch_size, True,False)

model = SNERModel(d_in=768, d_hid=100, d_class=len(dataset.cateDict) + 1, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)
lossFunc = nn.CrossEntropyLoss()


def evalTrainer():
    epochLoss = 0.0
    cycle = 0
    model.eval()
    for passage, mask, label in devLoader:
        passage = passage.long()
        emb = dataset.model(passage, attention_mask=mask)[0]

        out = model(emb)
        loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                        label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
        # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
        epochLoss += loss.item()
        cycle += 1
    return epochLoss


def trainTrainer(epoch):
    for i in trange(epoch):
        model.train()
        epochLoss = 0.0
        evalLoss = 10
        cycle = 0
        for passage, mask, label in trainLoader:
            passage = passage.long()
            emb = dataset.model(passage, attention_mask=mask)[0]

            out = model(emb)
            optimizer.zero_grad()
            loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                            label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
            loss.backward()
            optimizer.step()
            print(loss.item())
            epochLoss += loss.item()
            cycle += 1

        epochLoss = epochLoss / cycle
        evalLoss = min(evalTrainer(), evalLoss)

        print(epochLoss, evalLoss)


if __name__ == "__main__":
    trainTrainer(epoch)
