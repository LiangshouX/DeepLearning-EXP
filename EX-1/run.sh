#!/bin/bash

opt_values=("Adam" "SGD" "Adagrad" "Adadelta" "RMSprop")
loss_values=("CrossEntropyLoss "NLLLoss)

# 遍历
for opt in "${opt_values[@]}"
do
    for loss in "${loss_values[@]}"
    do
        # 使用当前的优化函数和损失函数运行Python脚本
        python3 Exp1-HandWritingNumRec.py -opt "$opt" -loss "$loss"
    done
done
