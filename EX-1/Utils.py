"""
    对所写代码的一些功能的测试

"""

import os

import cv2
import cv2 as cv
import torch
import numpy as np
import logging

def knowTheData():
    img = cv.imread("./dataset/train/images/train_1.bmp")
    print(type(img))
    print(img.shape)    # (28, 28, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)


def readLabels(label_path="./dataset/train/labels_train.txt"):
    """读取标签的txt文件并将其转换为PyTorch张量"""
    with open(label_path, 'rb') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        # row = [int(x) for x in line.strip().split()]
        row = int(line.strip())
        data.append(row)
    # print(type(lines), lines)
    print("Len(data):", len(data), '\t', "type(data):", type(data))
    # print("Len(data[0]):", len(data[0]), '\t', "type(data[0]):", type(data[0]))
    tensor = torch.tensor(data)
    return tensor


def readBmpFiles(img_path):
    """读取bmp文件并将其转换为PyTorch张量"""
    with open(img_path,'rb') as f:
        bmp_data = f.read()
    img = cv.imdecode(np.frombuffer(bmp_data, np.uint8), cv.IMREAD_COLOR)
    return torch.from_numpy(img).permute(2,0,1).float() / 255.0

def filterFile(directoryPath, extension="bmp"):
    """函数功能为过滤出给定目录下所选扩展名的文件
    Args:
        directoryPath(str):文件所在目录的相对路径
        extension(str):文件扩展名
    Returns:
        过滤的文件列表
    """
    relevant_path = directoryPath
    included_extensions = [extension]
    file_names = [file1 for file1 in os.listdir(relevant_path) if
                  any(file1.endswith(ext) for ext in included_extensions)]
    numberOfFiles = len(file_names)
    listParams = [file_names, numberOfFiles]
    return listParams

def try_log():
    filename = "./log/test.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=filename, encoding='utf-8')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    # 记录日志信息
    for i in range(10):
        logger.info("这是第\t%d\t条日志"%i)

if __name__ == '__main__':
    # knowTheData()

    # a = readBmpFiles("./dataset/train/images/train_1.bmp")
    # print(type(a), '\t', a.shape, '\n', a)

    # b = readLabels()
    # print(type(b), '\t', b.shape, '\n', b)
    try_log()
