# evaluate

模型的输出是一个（l × l × c）的打分矩阵，对每一个起止为i和j的spans输出其在输入各个实体类别的概率分布，需要将其转化为和原始输入数据相类似的实体条目信息，方便在验证集上计算F1分数，并可以实现完整的解码过程，得到实体输出列表。

#### 解码流程

argmax：对输出矩阵的最后一个维度求argmax，得到预测类别，并保存对于每个位置类别预测的分数情况。直接使用max操作得到预测类别矩阵和打分矩阵。

```python
tensor.max(dim = -1)
```

check_non-entity & check_clash：如果某个span预测得到的类别是0即为non-entity类别，将其在打分矩阵的对应分数mask为0；如果某两个实体之间存在交叠，即实体与实体的起止位置存在冲突，则只保留在打分矩阵中得分较高的预测实体，利用打分矩阵来构造mask。处理完成后应该会得到一个非常稀疏的预测类别矩阵。（目前的实现方法可能存在问题）

nonzero：将check完成后的稀疏矩阵中的非零元素取出，构造元组列表，一个元组包含一个实体的基本信息（起止位置+类别id），真实的label矩阵和check后的预测概率矩阵都需要抽取非零实体，然后比对其中相同元组，以计算F1分数。

#### 当前问题：

f1值都是0，问题很可能是因为没有排除无标签的实体。回去先检查一下argmax后的矩阵是否有不为零的类别标签，确认可预测出标签后，利用mask，把所有category为0的pred概率置为0.

目前发现模型趋向于对所有spans输出类别0，即无类别标签，也可能是训练不充分，模型容量太小。

最新实验结果可产生很小的f1分数值，可能是解码过程有点问题，且有个问题是对输出矩阵的最后一个维度没有做softmax

### 新的解码思路 Rm2entities

借鉴自sota论文的解码思路

https://github.com/juntaoy/biaffine-ner/blob/master/biaffine_ner_model.py

抽取出所有非零entity， (pos_s, pos_e, cate, score)

以score为key进行排序，得到top_spans列表，记录着所有entity

定义个抽取后的结果list，每次存入时不发生冲突即存入

实际实现时根据当前的代码逻辑做了一些调整，已存在对batch的循环，一次只需要对一个sentence做解码即可，先采用类似的过滤方法，抽取出所有不为0类别的实体，根据分数sort，循环抽取出不冲突的实体，格式转化为tuple的集合并返回，已经完成本地实现和测试，晚点儿放到服务器上运行

新的解码方法有效果了。
