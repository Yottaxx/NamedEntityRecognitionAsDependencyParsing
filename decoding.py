import torch
import os
import pandas as pd
import numpy as np
from utils.test.testdataset import TestDataset
from Model.FModel import FModel
from utils.util import Rm2entities, entities2df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_txt(path):
    sentence = pd.read_csv(path, names=['Text'])['Text']
    sentence = ' '.join(list(sentence))
    print("Text length: ", len(sentence))
    id = path.split('/')[-1].split('.')[0]
    return id, sentence

def data2dict(file_path):
    file_list = os.listdir(file_path)  # 3956
    text_dict = {}
    for i, filename in enumerate(file_list):
        path = file_path + '/' + filename
        text_id, text = read_txt(path)
        text_dict[int(text_id)] = text
    return text_dict # 3956

def load_model(d_h=150, n_l=2, dropout=0.5, path=None):
    model = FModel(d_in=1024, d_hid=d_h,
                   d_class=15, n_layers=n_l, dropout=dropout).to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def run():
    file_path = r"./utils/test/data/"
    label_path = r'./utils/test/label/'
    model_path = r'./checkpoint/2l-150h-4b-200e-roberta-finetune.pt'
    category_list = ['QQ', 'address', 'book', 'company', 'email',
                'game', 'government', 'mobile', 'movie',
                'name', 'organization', 'position', 'scene', 'vx']
    #texts = data2dict(file_path)
    #print(len(texts))
    test_dataset = TestDataset(path=file_path)
    id = 0
    for raw_text, input_ids, mask in test_dataset:
        # print("Text length", len(raw_text))
        # print(input_ids, len(input_ids[0]))
        # print(mask, len(mask[0]))
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        model = load_model(path=model_path)
        with torch.no_grad():
            out = model(input_ids, mask)
        out = out[0]
        entities = list(Rm2entities(out, is_flat_ner=False))
        new_entities = []
        for i, entity in enumerate(entities):
            pos_s, pos_e, category = entity
            pos_s = int(pos_s)
            pos_e = int(pos_e)
            privacy = raw_text[pos_s:pos_e+1]
            category = category_list[category-1]
            new_entities.append((str(category), str(pos_s), str(pos_e), str(privacy)))

        new_entities = sorted(new_entities, key=lambda x:x[1])
        df = entities2df(new_entities)
        df['ID'] = str(id)
        path = label_path + str(id) + '.csv'
        print("Saved " + path)
        id += 1
        df.to_csv(path, index=False)

def get_submission(label_path = r'./utils/test/label/', predict_path = None, count = 3956):
    path = r'./utils/test/label/'
    dfs = []
    for i in range(count):
        file_path = path+str(i)+'.csv'
        df = pd.read_csv(file_path)
        df = df.apply(lambda x:line2strs(x),axis=1)
        #print(df.iloc[0]['ID'], type(df.iloc[0]['ID']))
        dfs.append(df)
        #break
    res = pd.concat(dfs)
    if predict_path is None:
        predict_path = label_path+'predict.csv'
    res.to_csv(predict_path, index=False)
    print("Saved predict: ", predict_path)

def line2strs(line):
    #print(type(line))
    for i, v in line.items():
        #print(i, v)
        line[i] = str(v)
        #print(type(v))
    return line

if __name__ == '__main__':
    label_path = r'./utils/test/label/'
    predict_path = label_path+'predict_finetune_roberta.csv'
    run()
    get_submission(label_path=label_path, predict_path=predict_path)
