"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""
import os
from datetime import datetime

from tqdm import tqdm

from Exp3_Config import Training_Config
from Exp3_DataSet import TextDataSet, TestDataSet
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, epoch_no, epoch_num):
    """
    模型训练函数
    Args:
        model: 模型
        loader: train dataset loader
        epoch_no: 第几个epoch
        epoch_num: 一共多少epoch
    Returns:
        None
    """
    print('start training')
    model.train()
    total_loss = 0.0
    t = tqdm(loader)
    train_acc_num = 0.0
    for index, (data, label) in enumerate(t):
        # print('data')
        # print('label',label)
        if torch.cuda.is_available():
            label = label.cuda()
            for i in range(len(data)):
                try:
                    data[i] = data[i].cuda()
                except:
                    pass

        model.zero_grad()
        out = model(data)
        _, pred = torch.max(out, 1)
        # print('pred', pred)
        # print('label', label)
        num_correct = (pred == label).sum().item()
        train_acc_num += num_correct

        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
    train_avg_loss = total_loss / len(loader.dataset)
    print('Epoch: {}/{} \ttrain loss: {} \tacc: {}%'.format(epoch_no, epoch_num, train_avg_loss,
                                                            100 * train_acc_num / len(loader.dataset)))


def validation(model, loader, epoch):
    print('start validation')
    model.eval()
    eval_acc_num = 0
    global best_acc
    with torch.no_grad():
        t = tqdm(loader)
        for index, (data, label) in enumerate(t):
            if torch.cuda.is_available():
                label = label.cuda()
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            out = model(data)
            # print('out shape',out.shape)
            # print('out',out)
            _, pred = torch.max(out, 1)
            # print('pred',pred)
            # print('label',label)
            num_correct = (pred == label).sum().item()
            eval_acc_num += num_correct
    acc = 100 * eval_acc_num / len(loader.dataset)
    print("Validation Acc:\t{:.6f}%".format(acc))

    if acc > best_acc:
        print("Saving Model.....")
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc


def predict(model, loader):
    model.eval()

    # 加载checkpoint
    check_point = torch.load('checkpoint/ckpt.pth')
    model.load_state_dict(check_point['model'])
    optimizer.load_state_dict(check_point['optimizer'])

    print('start test')
    t = tqdm(loader)
    for index, data in enumerate(t):
        if torch.cuda.is_available():
            for i in range(len(data)):
                try:
                    data[i] = data[i].cuda()
                except:
                    pass
        out = model(data)
        _, pred = torch.max(out, 1)
        for i in pred:
            with open(config.test_result_path, 'a', encoding='utf-8') as f:
                f.write(str(i.cpu().numpy()))
                f.write('\n')
    # print("文件写入完毕，共{}条记录！".format(len(test_dataset)))


if __name__ == "__main__":
    config = Training_Config()

    # 训练集验证集
    train_dataset = TextDataSet(filepath=config.path_root + config.train_data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)
    # print(len(train_loader))
    val_dataset = TextDataSet(filepath=config.path_root + config.val_data_path)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    # 测试集数据集和加载器
    test_dataset = TestDataSet(config.path_root + config.test_data_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # 初始化模型对象
    Text_Model = TextCNN_Model()
    Text_Model = Text_Model.to(device)
    resume = True
    if resume:
        if not os.path.isdir(config.path_root + 'checkpoint'):
            print('Warning: no checkpoint directory found!')
            start_epoch = 0
            best_acc = 0.0
        else:
            # load model
            print("==> resume form checkpoint......")
            checkpoint = torch.load(config.path_root + './checkpoint/ckpt.pth')
            Text_Model.load_state_dict(checkpoint['model'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters())  # torch.optim中的优化器进行挑选，并进行参数设置

    msg = """
            +-------------------------------------------------------+
            |{:^47}|
            |=======================================================|
            |Optimizer: {:<10}  lr: {:<10}  Device: {}    |
            |-------------------------------------------------------|
            |Loss Function:{:<20}  Max_Epoch:{:<3}      |
            |=======================================================|
            |vocab size: {:<15}  batch size: {:<14}|
            |=======================================================|
            |训练集长度: {:<17}  测试集长度: {:<16}|
            |=======================================================|
            |{:<55}|
            +-------------------------------------------------------+
            """.format("模型设置以及代码运行环境", 'Adam', config.lr,
                       device,
                       str(loss_function), config.epoch, config.vocab_size, config.batch_size, len(train_dataset),
                       len(test_dataset), str(datetime.now()))
    print(msg)

    # 训练和验证
    print('start epoch', start_epoch)
    for i in range(start_epoch, start_epoch + config.epoch):
        train(Text_Model, loader=train_loader, epoch_no=i, epoch_num=config.epoch + start_epoch)
        if i % config.num_val == 0:
            validation(Text_Model, loader=val_loader, epoch=i)

    # 预测（测试）
    predict(Text_Model, test_loader)
