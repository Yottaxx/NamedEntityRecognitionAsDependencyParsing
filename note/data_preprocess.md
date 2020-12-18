# data_preprocess

数据预处理，将提供的data读入并转化为模型可以处理的模式。

#### 数据读入（loader）

将输入的txt文件和目标输出的csv文件读入，根据文件名一一对应，先读入结构较复杂的label为pandas的df，将txt储存的文本读入为str，储存到df的每一行中，df可看作一个数据点，其中每一行代表着每一个实体条目，且每一行都储存有文本信息。

一条信息的内容：dataID，entityID，category，pos_b，pos_e，entity。

#### 数据基本过滤（loader）

考虑到数据中可能会存在一些问题数据：

- pos和给定的entity不相符，对出现错误的df行删除；
- 对一个实体重复打标签的问题，当前模型框架不好处理，先忽略存在这类问题的数据点（id=30 & id=667）。

经过过滤的数据基本格式不变，每一个数据点依然用一个df来表示，整体数据集合是一个df的list。

#### 数据格式化（dataStruct）

将df的数据转化为struct格式化储存（也可用dict），避免重复储存文本数据，并形成封装类，可调用处理完成的数据以及categories字典等数据方面的信息。

进一步需要将输入文本和label格式化，这一步的代码在myDataset中。

#### 数据token化、批量化

输入的文本是str，需要将每一个汉字进行编码，采用预训练的XLnet模型对中文文本进行编码，预训练模型提供了汉语token词典，将中文字符转化为索引，索引可输入到预训练模型中经过前向计算得到字符的embedding。（MyDataset中处理）

输入的label信息只给出了每一条实体的起止位置和实体类别，抽象为edge tensor和value tensor，需要将其转换为l*l的矩阵，每一个第i行第j列的元素即代表了起止分别为ij的span的实体标签，0即为无实体标签。（BucketDataloader中处理）

每个数点都可以实现对应的转化，接下来需要将数据封装为torch中提供的Dataset类，并padding构造minibatch以方便输入到模型中。（这部分代码理解还不够深入）

已经批量化的数据集，可直接作为可迭代对象，迭代取出输入到模型中。将整个数据集3：1分割为训练集和验证集，只有训练集用于更新参数。

（发现label中会出现超过14的数值，因为个别的label中存在对同一个pos起始和终止位置的切片重复打标签的问题，先忽略这两个label。）

打印完整矩阵的方法

```python
torch.set_printoptions(profile="full")
print(x) # prints the whole tensor
torch.set_printoptions(profile="default")
```

