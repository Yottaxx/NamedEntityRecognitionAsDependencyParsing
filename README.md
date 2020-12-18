
### 2020年语言信息处理

# NER/命名实体标注

## Datasets/数据集
[非结构化商业文本信息中隐私信息识别](https://www.datafountain.cn/competitions/472/datasets)

## Model/模型
[Named Entity Recognition as Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.577/)
## Pretrained/预处理
* transformers hfl/chinese-xlnet-mid
* clue/roberta_chinese_clue_large

## Environment/环境
* torch>=1.6.0
* torchtext==0.7.0
* pandas
* numpy
* tqdm
* transformers

## Hyperparameter Optimization/超参数优化
### 工具：NNI
#### Tuner/调优器
Tree-structured Parzen Estimator

### 可视化工具：tensorboard
#### 可视化训练过程loss和F1score变化曲线


#### Assessor/评估器
Curve Fitting Assessor 

## 可运行脚本
### 训练脚本：trainParser.py
#### 作为调参时可接受命令行传入的参数启动训练，可能需要先下载预训练模型，训练过程会记录tensorboard，并保存验证集上最优checkpoint。
### 解码脚本: decoding.py
#### 根据提供的模型checkpoint解码test数据集，返回每个text的解码结果，并合并为比赛submission需要的提交格式。

## Member/成员
中
关
村
路


