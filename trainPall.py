from tqdm import trange
import os
from Model.FModel import FModel
from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
from utils.scoreF1 import batch_computeF1
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 100
batch_size = 4
dataset = MyDataset(path="./utils/train/", count=2515)
trainLoader = BucketDataLoader(dataset, batch_size, True, True)
devLoader = BucketDataLoader(dataset, batch_size, True, False)

model = FModel(d_in=768, d_hid=1024, d_class=len(dataset.cateDict) + 1, n_layers=4, dropout=0.5)
model = model.cuda()
model = nn.DataParallel(model, device_ids=[0,6])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
lossFunc = nn.CrossEntropyLoss(reduction='mean')

ckp_path = os.path.join(os.getcwd(), 'checkpoint')


def timeSince(start_time):
    sec = time.time() - start_time
    min = sec // 60
    sec = sec % 60
    return "{} min {} sec".format(int(min), int(sec))


def evalTrainer():
    epochLoss = 0.0
    cycle = 0
    f1_score = 0
    model.eval()
    for passage, mask, label in devLoader:
        passage = passage.long()
        passage = passage.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if (len(passage.shape) < 2):
            passage = passage.unsqueeze(0)
            mask = mask.unsqueeze(0)

        out = model(passage, mask)
        loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                        label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
        # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
        epochLoss += loss.item() / batch_size
        cycle += 1
        f1_score += batch_computeF1(label, out)
        # print(f1_score)
    return epochLoss / cycle, f1_score / cycle


def trainTrainer(epoch):
    evalLoss = 9999
    for i in trange(epoch):
        print()
        start_time = time.time()
        model.train()
        epochLoss = 0.0
        cycle = 0
        for passage, mask, label in trainLoader:
            # print("-------------Training--------")
            passage = passage.long()
            passage = passage.to(device)
            mask = mask.to(device)
            label = label.to(device)

            # print(passage.shape)
            if (len(passage.shape) < 2):
                passage = passage.unsqueeze(0)
                mask = mask.unsqueeze(0)

            out = model(passage, mask)
            print("-----out--------",out.shape)
            # print("-------------modeling--------")
            optimizer.zero_grad()
            loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                            label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
            loss.backward()
            optimizer.step()
            # print("-------------lossing--------")
            # print(loss.item())
            epochLoss += loss.item() / batch_size
            cycle += 1

        epochLoss = epochLoss / cycle
        evalLoss_new, f1_score = evalTrainer()
        if evalLoss_new < evalLoss:
            evalLoss = evalLoss_new
            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': evalLoss_new,
                }, os.path.join(ckp_path, '100e_12b_1024h_0001lr_4l_5dp.pt')  # parser版本可根据参数情况来设置ckp文件名
            )
        # evalLoss = min(evalLoss_new, evalLoss)

        print("====Epoch: {} epoch_loss: {} dev_loss: {} F1 score: {}".format(i + 1, epochLoss, evalLoss, f1_score))
        print("    Time used: {}".format(timeSince(start_time)))


if __name__ == "__main__":
    trainTrainer(epoch)
    # evalTrainer()