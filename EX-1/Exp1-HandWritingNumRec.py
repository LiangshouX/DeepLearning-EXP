import logging
import os
import cv2
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

# 命令行参数设置
parser = argparse.ArgumentParser(description="Settings of the whole model")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="学习率")
parser.add_argument("-opt", "--optimizer", type=str, choices=['Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop'],
                    default='SGD', help="优化器")
parser.add_argument("-loss", "--loss_function", type=str, choices=['CrossEntropyLoss', 'MSELoss', 'L1Loss', 'NLLLoss',
                                                                   'BCELoss', 'BCEWithLogitsLoss', 'SoftMarginLoss'],
                    default="CrossEntropyLoss", help="损失函数")
opt_list = ['Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop']
loss_list = ['CrossEntropyLoss', 'MSELoss', 'L1Loss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss', 'SoftMarginLoss']

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print("代码运行环境：\t", device)

class HandWritingNumberRecognize_Dataset(Dataset):
    """对包含若干个独立图像数据的数据集数据加载
    Args:
        data_root(str):数据的根目录
        label_name(str or None):标签txt文件的名字
    Returns:
        PyTorch数据张量
    """

    def __init__(self, data_root, label_name):
        # 这里添加数据集的初始化内容
        self.data_root = data_root
        self.images = []
        self.labels = []
        # 读取文件夹中的图片和标签
        # 图片
        file_list = os.listdir(os.path.join(self.data_root, "images"))
        sorted_list = sorted(file_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for fileName in sorted_list:
            img = cv2.imread(os.path.join(self.data_root, "images", fileName))
            self.images.append(img)
        # 标签
        if not label_name:
            self.labels = [0 for i in range(len(self.images))]
        else:
            with open(os.path.join(self.data_root, label_name), 'r') as f:
                lines = f.readlines()
            for line in lines:
                row = int(line.strip())
                self.labels.append(row)

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        img = self.images[index]
        label = self.labels[index]
        # 图片转换成PyTorch张量
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, label

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.images)


class HandWritingNumberRecognize_Network(nn.Module):
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        # self.fc1 = nn.Linear(in_features=28*28*3, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # self.fc3 = nn.Linear(in_features=512, out_features=128)
        # self.dropout1 = nn.Dropout(0.5)
        # self.fc4 = nn.Linear(in_features=128, out_features=10)
        self.fc1 = nn.Linear(in_features=28 * 28 * 3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # 此处添加模型前馈函数的内容，return函数需自行修改
        x = x.view(-1, 28 * 28 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = self.fc4(x)
        return x


def validation():
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    model.eval()

    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for i, data in enumerate(data_loader_val, 0):
            # 在这一部分撰写验证的内容，下面两行不必保留
            images, true_labels = data[0].to(device), data[1].to(device)
            total += len(true_labels)
            outputs = model(images)
            # _, pred_label = outputs.max(1)
            _, pred_label = torch.max(outputs.data, 1)
            # if i == 400:
            #     print(type(pred_label), '\n', pred_label, '\n', true_labels)

            # if pred_label == true_labels:
            #     correct += 1
            correct += (pred_label == true_labels).sum().item()
    accuracy = correct / total * 100.
    # print("验证集数据总量：", total, "预测正确的数量：", correct, end='\t')
    # print("当前模型在验证集上的准确率为：{:.3f}%".format(accuracy))
    return [total, correct, accuracy]


def alltest(opt_name, loss_name):
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    result_root = "./predict_results"
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    result_file_name = opt_name + "-" + loss_name + "-res.txt"

    model.eval()
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader_test, 0):
            images = data[0].to(device)
            total += len(images)
            outputs = model(images)
            _, pred_label = torch.max(outputs.data, 1)

            with open(result_root + '/' + result_file_name, 'a') as f:
                for res in pred_label:
                    f.write(str(res.cpu().item()) + '\n')
    print("\n test finished! %d test images in total." % total)


def train(epoch_num):
    # 循环外可以自行添加必要内容
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for index, data in enumerate(data_loader_train, 0):
        # print(data[1].tolist())
        images, true_labels = data[0].to(device), data[1].to(device)
        # 该部分添加训练的主要内容
        # 必要的时候可以添加损失函数值的信息，即训练到现在的平均损失或最后一次的损失，下面两行不必保留
        if index == len(data_loader_train) - 1 and len(images) < batch_size:
            total += len(images)
        else:
            total += batch_size
        # 梯度归零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # print(outputs.tolist())
        # print(outputs.shape)    # [64, 10]
        # print(true_labels.shape)    # [64]
        # _, predictd = outputs.max(1)
        # print(predictd.shape)      # [64]

        # 计算损失
        loss = loss_function(outputs, true_labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 统计训练信息
        train_loss += loss.item()
        _, predictd = outputs.max(1)
        correct += predictd.eq(true_labels).sum().item()
    # print(len(data_loader_train))
    train_loss /= len(data_loader_train)
    # print(total)  # 39923
    train_acc = 100. * correct / total
    return [train_loss, train_acc]


def save_log(file_name, msg):
    filename = file_name
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=filename, encoding='utf-8')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    # 记录日志信息
    logger.info(msg)


if __name__ == "__main__":
    args = parser.parse_args()

    # 设置超参数及损失函数和优化器
    batch_size = 64
    lr = args.learning_rate
    opt = args.optimizer
    loss = args.loss_function

    # 构建数据集，参数和值需自行查阅相关资料补充。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_root = "./dataset/train"
    train_label = "labels_train.txt"
    dataset_train = HandWritingNumberRecognize_Dataset(train_root, train_label)

    val_root = "./dataset/val"
    val_label = "labels_val.txt"
    dataset_val = HandWritingNumberRecognize_Dataset(val_root, val_label)

    test_root = "./dataset/test"
    test_label = None
    dataset_test = HandWritingNumberRecognize_Dataset(test_root, test_label)

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network().to(device)

    # 损失函数设置
    # loss_list = ['CrossEntropyLoss', 'MSELoss', 'L1Loss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss', 'SoftMarginLoss']
    loss_index = loss_list.index(loss)  # torch.nn中的损失函数进行挑选，并进行参数设置
    if loss_index == 0:
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss_index == 1:
        loss_function = torch.nn.MSELoss()
    elif loss_index == 2:
        loss_function = torch.nn.L1Loss()
    elif loss_index == 3:
        loss_function = torch.nn.NLLLoss()
    elif loss_index == 4:
        loss_function = nn.BCELoss()
    elif loss_index == 5:
        loss_function = nn.BCEWithLogitsLoss()
    elif loss_index == 6:
        loss_function = nn.SoftMarginLoss()
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        print("损失函数设置不正确，请重新设置："
              "['CrossEntropyLoss', 'MSELoss', 'L1Loss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss', 'SoftMarginLoss']")
        exit(1)

    # 优化器设置
    # opt_list = ['Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop']
    opt_index = opt_list.index(opt)
    if opt_index == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_index == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_index == 2:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif opt_index == 3:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif opt_index == 4:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = None  # torch.optim中的优化器进行挑选，并进行参数设置
        print("优化器选择不正确，请重新选择：['Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop']")
        exit(2)

    max_epoch = 15  # 自行设置训练轮数
    num_val = 3  # 经过多少轮进行验证

    msg = """
        +-------------------------------------------------------+
        |{:^47}|
        |=======================================================|
        |Optimizer: {:<10}  lr: {:<10}  Device: {}  |
        |-------------------------------------------------------|
        |Loss Function:{:<20}  Max_Epoch:{:<3}      |
        |=======================================================|
        |{:<55}|
        +-------------------------------------------------------+
            """.format("模型设置以及代码运行环境", args.optimizer, args.learning_rate, device, args.loss_function,
                       max_epoch, str(datetime.datetime.now()))
    print(model)
    print(msg)
    # exit(1)

    log_root = "./log"
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    log_file_name = log_root + '/' + "train-" + str(args.optimizer) + '-' + str(args.loss_function) + '.log'
    save_log(log_file_name, msg)

    # exit(1)

    # 然后开始进行训练
    for epoch in range(max_epoch):
        [train_loos, train_acc] = train(epoch)
        # 在训练数轮之后开始进行验证评估
        if (epoch + 1) % num_val == 0:
            [val_total, val_correct, val_acc] = validation()
            train_msg = "epoch[{}/{}], train_loss:{:.3f},train_acc:{:.2f}, " \
                        "val_total:{}, val_correct:{}, val_acc:{:.2f}". \
                format(epoch + 1, max_epoch, train_loos, train_acc, val_total, val_correct, val_acc)
            save_log(log_file_name, train_msg)
            print(train_msg)
        else:
            train_msg = "epoch[{}/{}], train_loss:{:.3f},train_acc:{:.2f}". \
                format(epoch + 1, max_epoch, train_loos, train_acc)
            save_log(log_file_name, train_msg)
            print(train_msg)

        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, 'model_checkpoint.pth')
    # 自行完善测试函数，并通过该函数生成测试结果
    alltest(args.optimizer, args.loss_function)
