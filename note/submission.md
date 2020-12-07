# submission record

所有实验结果截图均放在Experiment_result文件夹下，包含三个模型的训练过程记录（利用tensorboard保存为SVG位图）、test提交结果截图、predict提交文件。另附上三次实验结果的对比图（三张）。

### Xlnet

参数设置：

- LSTM_layers = 2
- Input_nodes(embedding_dims) = 768
- Hidden_nodes = 150
- lr = 0.001
- dropout = 0.5
- batch_size = 16
- epoch_nums = 200

保存了最佳F1score的模型，采用nested解码（即可以解码出嵌套实体），后续均沿用模型保存方法和解码方法。

模型不参加训练，即没有微调XLnet模型。

### Roberta

参数设置：

- LSTM_layers = 2
- Input_nodes(embedding_dims) = 1024
- Hidden_nodes = 150
- lr = 0.001
- dropout = 0.5
- batch_size = 64
- epoch_nums = 200

同样保存最佳F1score的模型，nested解码。模型不参加训练。

相对于XLnet载训练集和验证集上的F1分数都有显著上升，且test提交分数可超过0.70，但观察到较为明显的过拟合情况。

### Roberta_finetune

参数设置：

- LSTM_layers = 2
- Input_nodes(embedding_dims) = 1024
- Hidden_nodes = 150
- lr = 0.0001
- Roberta_finetune_lr = 2e-5
- dropout = 0.3
- batch_size = 4
- epoch_nums = 200

同样保存最佳F1score的模型，nested解码设置，模型参加微调。可并行化的batch大小显著下降。

相对于不微调的Roberta模型在test提交分数上有2个百分点的提升，且训练过程没有出现明显的过拟合（loss存在波动，可能需要更小的lr）。