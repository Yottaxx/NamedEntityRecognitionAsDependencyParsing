#-*- coding: UTF-8 -*-

import argparse
import os

#import nni
from tqdm import trange
import time

from Model.FModel import FModel
from Model.SModel import SModel
from Model.SNERModel import SNERModel
from utils import MyDataset, BucketDataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, XLNetModel
import logging
from torch.utils.tensorboard import SummaryWriter

from utils.util import batch_computeF1, get_useful_ones

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger('NER')

path_prefix = r'/home/mgliu/work_NER/NewNER/checkpoint'


def get_paras():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--lr', '-l', type=float, help="lr must", default=0.001)
    parser.add_argument('--batch_size', '-b', type=int, help="batch_size must", default=16)
    parser.add_argument('--epoch', '-e', type=int, help="epoch must", default=200)
    parser.add_argument('--dropout', '-d', type=float, help="dropout must", default=0.5)
    parser.add_argument('--d_in', '-i', type=int, help="in_size must", default=768)
    parser.add_argument('--d_hid', '-g', type=int, help="g_size must", default=150)
    parser.add_argument('--n_layers', '-k', type=int, help="kernel must", default=2)
    parser.add_argument('--redo', '-r', type=int, help="reload model", default=0)

    args, _ = parser.parse_known_args()
    return args


def load_dict(model, path, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("load: epoch ", epoch + " loss " + loss)
    return model, optimizer


def save_dict(model, path, optimizer, epoch, loss):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(path_prefix, path)  # parser版本可根据参数情况来设置ckp文件名
    )
    print("++++Checkpoint Saved at :", os.path.join(path_prefix, path))


def run(args):
    epoch = args['epoch']
    batch_size = args['batch_size']

    dataset = MyDataset(path="./utils/train/", count=2515)
    trainLoader = BucketDataLoader(dataset, batch_size, True, True)
    devLoader = BucketDataLoader(dataset, batch_size, True, False)

    model = FModel(d_in=args['d_in'], d_hid=args['d_hid'],
                   d_class=len(dataset.cateDict) + 1, n_layers=args['n_layers'], dropout=args['dropout']).to(device)
    for name, param in model.named_parameters():
        if "model" in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), weight_decay=5e-4)
    if args['redo'] == 1:
        model, optimizer = load_dict(model, os.path.join(args["n_layers"] + args["d_hid"] + args["batch_size"]),
                                     optimizer)

    #model = nn.DataParallel(model)
    lossFunc = nn.CrossEntropyLoss(reduction='sum')

    def timeSince(start_time):
        sec = time.time() - start_time
        min = sec // 60
        sec = sec % 60
        return "{} min {} sec".format(int(min), int(sec))

    def evalTrainer():
        epochLoss = 0.0
        cycle = 0
        Fscore = 0.0
        precision = 0.0
        recall = 0.0
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
            tmp_out, tmp_label = get_useful_ones(out, label, mask)
            # loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
            #                 label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
            loss = lossFunc(tmp_out, tmp_label)
            # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
            epochLoss += loss.item()
            cycle += 1
            Fscore_tmp, precision_tmp, recall_tmp = batch_computeF1(label, out, mask)
            Fscore+=Fscore_tmp
            precision+=precision_tmp
            recall+=recall_tmp

        epochLoss = epochLoss / cycle
        Fscore = Fscore / cycle
        precision = precision / cycle
        recall = recall / cycle
        # nni.report_intermediate_result(epochLoss)
        return epochLoss, Fscore, precision, recall

    def trainTrainer(epoch):
        evalLoss = 9999
        evalFscore = 0.0
        evalP = 0.0
        evalR = 0.0
        precision = 0.0
        recall = 0.0
        for i in trange(epoch):
            print()
            start_time = time.time()
            model.train()
            epochLoss = 0.0
            Fscore = 0.0
            # evalLoss = 9999
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

                # print("-------------embing--------")
                out = model(passage, mask)
                # print("-------------modeling--------")
                optimizer.zero_grad()
                tmp_out, tmp_label = get_useful_ones(out, label, mask)
                # loss = lossFunc(out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1),
                #                 label.reshape(label.shape[0] * label.shape[1] * label.shape[2]))
                loss = lossFunc(tmp_out, tmp_label)
                # loss = -(c * torch.log(F.softmax(out, dim=-1))).sum()
                loss.backward()
                optimizer.step()
                # print("-------------lossing--------")
                # print(loss.item())
                epochLoss += loss.item()
                cycle += 1
                Fscore_tmp, precision_tmp, recall_tmp = batch_computeF1(label, out, mask)
                Fscore += Fscore_tmp
                precision += precision_tmp
                recall += recall_tmp

            epochLoss = epochLoss / cycle
            precision = precision / cycle
            recall = recall / cycle
            evalTrainerLoss, evalTrainerF1, evalTrainerP, evalTrainerR = evalTrainer()

            if evalTrainerF1 > evalFscore:
                save_dict(model,
                          os.path.join(str(args["n_layers"]) +'l-'+ str(args["d_hid"]) +'h-'+ str(args["batch_size"])+'b-'
                                       +"200e-transformer.pt"),
                          optimizer, i, evalLoss)
            if evalTrainerLoss < evalLoss:
                save_dict(model,
                          os.path.join(str(args["n_layers"]) +'l-'+ str(args["d_hid"]) +'h-'+ str(args["batch_size"])+'b-'
                                       +"200e-minloss_transformer.pt"),
                          optimizer, i, evalLoss)

            evalLoss = min(evalTrainerLoss, evalLoss)
            evalFscore = max(evalFscore, evalTrainerF1)
            evalP = max(evalP, evalTrainerP)
            evalR = max(evalR, evalTrainerR)
            Fscore = Fscore / cycle

            writer.add_scalar('Loss/train', epochLoss, i)
            writer.add_scalar('Loss/test', evalTrainerLoss, i)
            writer.add_scalar('Accuracy/train', Fscore, i)
            writer.add_scalar('Accuracy/test', evalTrainerF1, i)
            writer.add_scalar("Precision/train", precision, i)
            writer.add_scalar("Precision/test", evalTrainerP, i)
            writer.add_scalar("Recall/train", recall, i)
            writer.add_scalar("Recall/test", evalTrainerR, i)

            print("====Epoch: {} epoch_F: {} dev_F: {}".format(i + 1, Fscore, evalTrainerF1))
            print("    Time used: {}".format(timeSince(start_time)))
        # nni.report_final_result(evalLoss)

    trainTrainer(epoch)


# TODO(Yotta): to evaluation tasks

if __name__ == '__main__':
    try:
        #tuner_params = nni.get_next_parameter()
        #logger.debug(tuner_params)
        params = vars(get_paras())
        #params.update(tuner_params)
        print(params)
        run(params)
    except Exception as exception:
        logger.exception(exception)
        raise
