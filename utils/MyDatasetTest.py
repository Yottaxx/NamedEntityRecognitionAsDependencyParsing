from torchtext import data
import torch
from torch.nn import init

from Model.SModel import SModel
# from transformers import AutoTokenizer, XLNetModel
#
# from Model.SNERModel import SNERModel
# from utils.dataStruct import MyDataLoader
#
# # def tokenizer(text):
# #     return [tok for tok in text]
#
#
# dataset = MyDataLoader(count=10).loadData()
# print(dataset[0].text)
#
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-mid")
#
# model = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid", mem_len=768, return_dict=True)
#
# text = [token for token in dataset[0].text[0]]
#
# print(text)
# encoding = tokenizer.encode_plus(text, return_tensors='pt')
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']
# outputs = model(input_ids, attention_mask=attention_mask)
#
# NERModel = SNERModel(d_in=768, d_hid=100, d_class=12, n_layers=2)
# out = NERModel(outputs[0])

# edgeLeft = data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
# edgeRight = data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
# values = data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
# text = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer, lower=True)
# train = data.Dataset(dataset, fields=[('edgeLeft', None),
#                                       ('edgeRight', None), ('values', None), ('text', text)])
#
# text.build_vocab(train, vectors='glove.6B.100d')  # , max_size=30000)
# text.vocab.vectors.unk_init = init.xavier_uniform
# len_vocab = len(text.vocab)
# print(text.vocab.vectors.shape)
#
# train_iter = data.BucketIterator(train, batch_size=1, train=True,
#                                  sort_within_batch=True,
#                                  sort_key=lambda x: (len(x.text)), repeat=False,
#                                  device='cpu')
#
# model = SModel(len_vocab, d_emb=100, d_in=100, d_hid=100, d_class=12, n_layers=2)
# model.embedding.weight.data.copy_(text.vocab.vectors)
#
# for batch in train_iter:
#     out = model(batch.text)
#     print(batch.text)
#     print(out)
