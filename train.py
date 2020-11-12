from tqdm import trange

from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 500
batch_size = 8
dataset = MyDataset(path="./utils/train/", count=2515)
trainLoader = BucketDataLoader(dataset, batch_size, True,True)
devLoader = BucketDataLoader(dataset, batch_size, True,False)

model = SNERModel(d_in=768, d_hid=768, d_class=len(dataset.cateDict) + 1, n_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)
lossFunc = nn.CrossEntropyLoss(reduction='mean')

#model_path = r'C:\Users\86435\Documents\work_pycharm\work_NER\chinese-xlnet-mid'
pretrained_model = XLNetModel.from_pretrained(dataset.model_path, mem_len=768).to(device).eval()

def timeSince(start_time):
    sec = time.time() - start_time
    min = sec//60
    sec = sec%60
    return "{} min {} sec".format(int(min), int(sec))

def evalTrainer():
    epochLoss = 0.0
    cycle = 0
    model.eval()
    for passage, mask, label in devLoader:
        passage = passage.long()
        passage = passage.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if (len(passage.shape) < 2):
            passage = passage.unsqueeze(0)
            mask = mask.unsqueeze(0)

        with torch.no_grad():
            emb = pretrained_model(passage, attention_mask=mask)[0]
            #emb = emb.to(device)

            out = model(emb)
            loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                        label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
        # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
        epochLoss += loss.item()/batch_size
        cycle += 1
    return epochLoss/cycle


def trainTrainer(epoch):
    for i in trange(epoch):
        print()
        start_time = time.time()
        model.train()
        epochLoss = 0.0
        evalLoss = 9999
        cycle = 0
        for passage, mask, label in trainLoader:
            #print("-------------Training--------")
            passage = passage.long()
            passage = passage.to(device)
            mask = mask.to(device)
            label = label.to(device)

            #print(passage.shape)
            if(len(passage.shape)<2):
                passage = passage.unsqueeze(0)
                mask = mask.unsqueeze(0)

            with torch.no_grad():
                emb = pretrained_model(passage, attention_mask=mask)[0]
            #emb = emb.to(device)

            #print("-------------embing--------")
            out = model(emb)
            #print("-------------modeling--------")
            optimizer.zero_grad()
            loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                            label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
            loss.backward()
            optimizer.step()
            #print("-------------lossing--------")
            #print(loss.item())
            epochLoss += loss.item()/batch_size
            cycle += 1

        epochLoss = epochLoss / cycle
        evalLoss = min(evalTrainer(), evalLoss)

        print("====Epoch: {} epoch_loss: {} dev_loss: {}".format(i+1, epochLoss, evalLoss))
        print("    Time used: {}".format(timeSince(start_time)))


if __name__ == "__main__":
    trainTrainer(epoch)
