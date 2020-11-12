import argparse

import nni
from tqdm import trange
import time

from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger('NER')


def get_paras():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--lr', '-l', type=float, help="lr must", default=0.001)
    parser.add_argument('--batch_size', '-b', type=int, help="batch_size must", default=32)
    parser.add_argument('--epoch', '-e', type=int, help="epoch must", default=16)
    parser.add_argument('--dropout', '-d', type=float, help="dropout must", default=0.3)
    parser.add_argument('--d_in', '-i', type=int, help="in_size must", default=768)
    parser.add_argument('--d_hid', '-g', type=int, help="g_size must", default=4 * 768)
    parser.add_argument('--n_layers', '-k', type=int, help="kernel must", default=2)
    parser.add_argument('--redo', '-r', type=int, help="reload model", default=0)

    args, _ = parser.parse_known_args()
    return args


def run(args):
    epoch = args['epoch']
    batch_size = args['batch_size']

    dataset = MyDataset(path="./utils/train/", count=2515)
    trainLoader = BucketDataLoader(dataset, batch_size, True, True)
    devLoader = BucketDataLoader(dataset, batch_size, True, False)

    model = SNERModel(d_in=args['d_in'], d_hid=args['d_hid'],
                      d_class=len(dataset.cateDict) + 1, n_layers=args['n_layers'], dropout=args['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), weight_decay=5e-4)
    lossFunc = nn.CrossEntropyLoss()

    pretrained_model = XLNetModel.from_pretrained(dataset.model_path, mem_len=768).to(device).eval()

    def timeSince(start_time):
        sec = time.time() - start_time
        min = sec // 60
        sec = sec % 60
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
                # emb = emb.to(device)

                out = model(emb)
                loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                                label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
            epochLoss += loss.item() / batch_size
            cycle += 1
        epochLoss = epochLoss/cycle
        nni.report_intermediate_result(epochLoss)
        return epochLoss

    def trainTrainer(epoch):
        evalLoss = 9999
        for i in trange(epoch):
            print()
            start_time = time.time()
            model.train()
            epochLoss = 0.0
            #evalLoss = 9999
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

                with torch.no_grad():
                    emb = pretrained_model(passage, attention_mask=mask)[0]
                # emb = emb.to(device)

                # print("-------------embing--------")
                out = model(emb)
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
            evalLoss = min(evalTrainer(), evalLoss)

            print("====Epoch: {} epoch_loss: {} dev_loss: {}".format(i + 1, epochLoss, evalLoss))
            print("    Time used: {}".format(timeSince(start_time)))
        nni.report_final_result(evalLoss)
    trainTrainer(epoch)


# TODO(Yotta): to evaluation tasks

if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_paras())
        params.update(tuner_params)
        print(params)
        run(params)
    except Exception as exception:
        logger.exception(exception)
        raise
