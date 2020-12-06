# run model

在/work_NER/NewNER下调用train.py，参数需要在脚本中设定。训练过程不会对XLnet预训练模型进行微调训练，每一个epoch中，选择3/4的数据用于训练，训练结束后调用eva函数进行验证，并对所有的epoch保留最小的dev_loss和最优模型的模型参数。

```bash
CUDA_VISIBLE_DEVICES='1' CUDA_LAUNCH_BLOCKING=1  python -u train.py > ../log/test.log 2>&1 &

```

训练过程的交叉熵损失默认为mean模式，但平均的结果会导致loss数值较小，可考虑使用sum模式，但同时梯度也会变大。

## nni调参

依然在/work_NER/NewNER，采用nni框架，调用train_parser.py来进行自动调参，train_parser.py会自动接受命令行传递的参数设置，对不同的参数设置进行固定轮次的训练。

```bash
# 启动nni调参，更改gpu设置时需要对config文件做修改
CUDA_VISIBLE_DEVICES='5' nnictl create --config NewNER/config.yml --port 8890

# 如果发生问题及时停止
nnictl stop
```

第一轮次较为粗糙的调参的搜索空间设置：

```json
{
    "epoch": {"_type":"choice", "_value": [64]},
    "batch_size": {"_type":"choice", "_value": [8,12]},
    "d_in":{"_type":"choice","_value":[768]},
    "d_hid":{"_type":"choice","_value":[384,512,768,1024]},
    "lr":{"_type":"choice","_value":[0.0001,0.005,0.001]},
    "dropout":{"_type":"choice","_value":[0.3,0.4,0.2,0.5]},
    "n_layers":{"_type":"choice","_value":[2,4,6]},
    "redo":{"_type":"choice","_value":[0]}
}
```

目前感觉可以改进的地方：epoch可以设置32，模型需要在较小的lr和dropout下才会有较好的表现，lstm的层数会较为显著地影响训练时长。

周末采用更为精细的搜索空间在进行一次搜索。考虑先解决f1的问题，如果模型只学习到了将所有的类别分类为0（non-entity），那就难搞了。。。

还是先尝试一次训练，看看随着epoch上升，f1分数是否有提升，并打印一些经过非零过滤的预测类别矩阵的打分矩阵，看看是否预测出了一些非0实体。

两个较为合适的训练条件：

```
/nni-experiments/toUYkbCF/trials/JESmp
```

![image-20201113231514673](C:\Users\86435\AppData\Roaming\Typora\typora-user-images\image-20201113231514673.png)

```
/nni-experiments/ggDTj89G/trials/tgx5U
```

![image-20201113232215046](C:\Users\86435\AppData\Roaming\Typora\typora-user-images\image-20201113232215046.png)

采用了上面第一组的参数设置，设置了100轮次的训练，但70+轮次之后整体F1分数依然是0，且每一个预测类别矩阵中，类别为0的概率依然是最大值，即经过max操作后，得到的所有预测类别都是0（无实体类别）。可能的原因：loss计算的问题，导致梯度总是使预测概率向0移动；解码问题，可考虑选择第二大概率的类别，用于解码。



