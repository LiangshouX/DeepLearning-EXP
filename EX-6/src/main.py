"""
    程序的主入口

"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

from utils import BLEUScore, blue_sore, plot_carve
from Dataset import E2EDataset
from Model import Encoder, Decoder, Seq2Seq
from Config import Config
import Model

PAD_ID = Model.PAD_ID

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss_lst, val_bleu_lst, test_attn_lst = [], [], []


def train(model, data_loader, epoch_current, epoch_total):
    """模型训练函数"""
    # print("Start Training...")
    model.train()
    total_loss = 0.0  # 打印输出的loss
    t1 = time.time()
    with tqdm(total=len(data_loader),
              desc='Training epoch[{}/{}]'.format(epoch_current, epoch_total),
              file=sys.stdout) as t:
        for index, batch_data in enumerate(data_loader):
            source, target = batch_data
            # print("\n src:\n", source, source.shape)     # (8, 40) -> [batch_size, seq_len]
            # print("target:\n", target, target.shape)    # (8, 40) -> [batch_size, seq_len]
            source, target = source.to(device).transpose(0, 1), target.to(device).transpose(0, 1)

            optimizer.zero_grad()  # 梯度值初始化
            # 前向传播
            model_outputs = model((source, target))
            # vocab_size = model_outputs.size()[-1]
            model_outputs = model_outputs.contiguous().view(-1, vocab_size)
            targets = target.contiguous().reshape(-1, 1).squeeze(1)
            # print("model_output shape:\t{}\n"
            #       "targets shape:\t{}\n".format(model_outputs.shape,
            #                                     targets.shape))

            loss = loss_function(model_outputs, targets.long())
            total_loss += loss.data.item()

            # 梯度下降
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=total_loss / (index + 1), lr=scheduler.get_last_lr()[0], timecost=time.time()-t1)
            t.update(1)
            # scheduler.step()
        loss_list.append(total_loss / len(data_loader))
        lr_list.append(scheduler.get_last_lr()[0])
        scheduler.step()
        # torch.cuda.empty_cache()


def validation(model, data_iterator, epoch_now):
    global best_bleu
    model.eval()
    sentences = []
    with torch.no_grad():
        for data in tqdm(data_iterator, desc="[Validation]{}".format(" "*(5+len(str(epoch_now)))), file=sys.stdout):
            src, tgt, lex, multi_target = data
            # print("\nsrc.size:\t{}".format(src.size()))
            src = torch.as_tensor(src[:, np.newaxis]).to(device)
            sentence, attention = model.predict(src)
            # 解码句子
            sentence = train_dataset.tokenizer.decode(sentence).replace('[NAME]', lex[0]).replace('[NEAR]', lex[1])
            sentences.append(sentence)
            scorer.append(sentence, multi_target)
        # print(sentences)
        # print(len(sentences))
        bleu = scorer.score()
        bleu_list.append(bleu)
        print("BLEU SCORE: {:.4f}".format(bleu))
        if bleu > best_bleu:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'bleu': bleu,
                'epoch': epoch_now,
            }
            if not os.path.exists(config.checkpoint_path):
                os.mkdir(config.checkpoint_path)
            torch.save(state, config.checkpoint_path + 'checkpoint.pth')
            print("模型保存成功！！")
            best_bleu = bleu
        val_bleu_lst.append(bleu)


def predict(model, data_loader):
    print("Start Prediction...")
    model.eval()
    if not os.path.exists(config.checkpoint_path):
        print("Warning: checkpoint directory not found!")
        exit(-2)
    else:
        # 加载模型
        print("===> Resume from checkpoint...")
        ckpt = torch.load(config.checkpoint_path + 'checkpoint.pth')
        model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        for data in tqdm(data_loader, desc="[TEST] {}", file=sys.stdout):
            src, tgt, lex, _ = data
            src = torch.as_tensor(src[:, np.newaxis]).to(device)
            src = src.to(device)
            # 模型预测
            sentence, attention = model.predict(src)
            sentence = train_dataset.tokenizer.decode(sentence).replace('[NAME]',
                                                                        lex[0]).replace('[NEAR]', lex[1])
            test_attn_lst.append(attention)
            # 写入文件
            with open(config.test_result_path, 'a+', encoding='utf-8') as f:
                f.write(sentence + '.\n')
    print("Predict Finished! Save result into {}".format(config.test_result_path))

def visualize_attention(dataset, data_index=0):
    """Attention可视化"""
    src, tgt, lex, _ = dataset[data_index]
    src = torch.as_tensor(src[np.newaxis, :]).to(device).transpose(0, 1)
    sentence, attention = model.predict(src)
    src_txt = list(map(lambda x: dataset.tokenizer.index_to_token(x),
                       src.flatten().cpu().numpy().tolist()[:10]))
    for i in range(len(src_txt)):
        if src_txt[i] == '[NAME]':
            src_txt[i] = lex[0]
        elif src_txt[i] == '[NEAR]':
            src_txt[i] = lex[1]
    sentence_txt = list(map(lambda x: dataset.tokenizer.index_to_token(x),
                            sentence))
    for i in range(len(src_txt)):
        if sentence_txt[i] == '[NAME]':
            sentence_txt[i] = lex[0]
        elif sentence_txt[i] == '[NEAR]':
            sentence_txt[i] = lex[1]

    # 绘制热力图
    ax = sns.heatmap(np.array(attention)[:, :10] * 100, cmap='YlGnBu')
    # 设置坐标轴
    plt.yticks([i + 0.5 for i in range(len(sentence_txt))], labels=sentence_txt, rotation=360, fontsize=12)
    plt.xticks([i + 0.5 for i in range(len(src_txt))], labels=src_txt, fontsize=12)
    plt.show()

if __name__ == "__main__":
    scorer = BLEUScore(max_ngram=4)
    trainSet_path = config.root_path + config.train_data_path
    devSet_path = config.root_path + config.dev_data_path
    testSet_path = config.root_path + config.test_data_path

    train_dataset = E2EDataset(trainSet_path, train_mod='train')

    dev_dataset = E2EDataset(devSet_path, train_mod='valid',
                             attributes_vocab=train_dataset.attributes_vocab,
                             tokenizer=train_dataset.tokenizer)

    test_dataset = E2EDataset(testSet_path, train_mod='test',
                              attributes_vocab=train_dataset.attributes_vocab,
                              tokenizer=train_dataset.tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # 初始化模型
    vocab_size = train_dataset.tokenizer.vocab_size
    # print(type(vocab_size), vocab_size)

    model = Seq2Seq(config=config,
                    device=device,
                    src_vocab_size=vocab_size,
                    tgt_vocab_size=vocab_size).to(device)
    best_bleu = 0.0
    loss_list = []
    bleu_list = []
    lr_list = []

    # 加载ckpt
    if not os.path.exists(config.checkpoint_path):
        print("Warning: checkpoint directory not found!")
        start_epoch = 0
        best_bleu = 0.0
    else:
        # 加载模型
        print("===> Resume from checkpoint...")
        checkpoint = torch.load(config.checkpoint_path + 'checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        best_bleu = checkpoint['bleu']
        start_epoch = checkpoint['epoch']

    # 设置损失函数和优化器
    weight = torch.ones(train_dataset.tokenizer.vocab_size)
    weight[PAD_ID] = 0
    loss_function = nn.NLLLoss(weight, reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
                """.format("模型设置以及代码运行环境", 'SGD', config.lr,
                           device,
                           str(loss_function), config.epoch, vocab_size, config.batch_size, len(train_dataset),
                           len(test_dataset), str(datetime.now()))
    print(msg)

    # 训练和验证
    print("Start Epoch ====>\t", start_epoch)
    for i in range(start_epoch, config.epoch):
        train(model, train_loader, i + 1, config.epoch)
        if (i + 1) % config.num_val == 0:
            validation(model, dev_dataset, i)
    plot_carve(title="valid_bleu", save_path="../res_img/valid_bleu.png",
               x=len(bleu_list), y=bleu_list)
    plot_carve(title="train_loss", save_path="../res_img/train_loss.png", x=len(loss_list), y=loss_list)
    plot_carve(title="train_lr", save_path="../res_img/train_lr.png", x=len(lr_list), y=lr_list)

    predict(model, test_dataset)

    visualize_attention(dev_dataset, data_index=10)
