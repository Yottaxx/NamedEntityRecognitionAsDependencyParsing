# submission record

![image-20201127152302007](C:\Users\86435\AppData\Roaming\Typora\typora-user-images\image-20201127152302007.png)

本周实验的一些结果，服务器上保存的checkpoint，分别对应的实验条件：

- 基础参数设置 predict-v0.1
- 同上(并没有引入真实的bfl)
- 引入了bfl后的ckpt predict-v0.2&3
- 引入bfl，并且根据minloss保存了模型 predict-v0.4
- 同上，实验中打印了PR信息 v0.5

predict-v0.3开始引入nest解码，较为有效地提升了分数，其余更新对结果改变不大。