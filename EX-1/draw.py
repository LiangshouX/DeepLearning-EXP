"""
    数据从log日志中收集。若需要做重复试验验证结果，请依次选择['Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop']作为优化器，
CrossEntropyLoss作为损失函数运行程序，并在log日志中查看相应的数据。
"""

import matplotlib.pyplot as plt

# 设置迭代次数和五个变量的训练准确率数据
epochs = [i for i in range(1, 16)]
var1_train_acc = [0.9032, 0.9631, 0.9787, 0.9875, 0.9924, 0.9958, 0.9978, 0.9990, 0.9994, 0.9997,
                  0.9998, 0.9998, 0.9999, 0.9999, 1.000]
var2_train_acc = [0.9060, 0.9502, 0.9602, 0.9644, 0.9678, 0.9701, 0.9694, 0.9712, 0.9728, 0.9726,
                  0.9746, 0.9757, 0.9768, 0.9782, 0.9793]
var3_train_acc = [0.9212, 0.9725, 0.9852, 0.9921, 0.9960, 0.9979, 0.9989, 0.9994, 0.9996, 0.9998,
                  0.9998, 0.9999, 0.9999, 1.000, 1.000]
var4_train_acc = [0.5947, 0.8431, 0.8875, 0.9098, 0.9240, 0.9340, 0.9410, 0.9468, 0.9523, 0.9571,
                  0.9603, 0.9641, 0.9668, 0.9694, 0.9720]
var5_train_acc = [0.8673, 0.9279, 0.9460, 0.9545, 0.9602, 0.9647, 0.9671, 0.9696, 0.9730, 0.9732,
                  0.9758, 0.9766, 0.9772, 0.9785, 0.9801]

# 创建折线图并添加数据
plt.plot(epochs, var1_train_acc, label='SGD')
plt.plot(epochs, var2_train_acc, label='Adam')
plt.plot(epochs, var3_train_acc, label='Adagrad')
plt.plot(epochs, var4_train_acc, label='Adadelta')
plt.plot(epochs, var5_train_acc, label='RMSprop')

# 添加标题、轴标签和图例
plt.title('Training Accuracy by Epoch and Optimizers')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend()

# 显示图形
plt.show()

# 保存图片
plt.savefig("Contrast.jpg")
