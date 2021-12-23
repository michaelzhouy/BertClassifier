# -*- coding: utf-8 -*-
# @Time    : 2021/12/11 10:49 上午
# @Author  : Michael Zhouy
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier, BertLstmClassifier
from dataset import CNewsDataset, CNewsDatasetDF
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'


def main():
    labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    # 参数设置
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 1
    learning_rate = 5e-6  # Learning Rate不宜太大
    MAX_LEN = 512
    # model_name_or_path = 'bert-base-chinese'
    # model_name_or_path = 'chinese_roberta_wwm_ext_pytorch'
    # model_name_or_path = 'chinese_xlnet_mid_pytorch'
    model_name_or_path = 'hfl/chinese-bert-wwm'
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    train = pd.read_table('./data/cnews/cnews.train.txt', header=None)
    valid = pd.read_table('./data/cnews/cnews.val.txt', header=None)
    train.columns = ['label', 'content']
    valid.columns = ['label', 'content']

    train['label'] = train['label'].map(lambda x: labels.index(x))
    valid['label'] = valid['label'].map(lambda x: labels.index(x))

    # 获取到dataset
    train_dataset = CNewsDatasetDF(train, tokenizer, MAX_LEN)
    valid_dataset = CNewsDatasetDF(valid, tokenizer, MAX_LEN)

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_name_or_path)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels)
    # model = BertLstmClassifier(bert_config, num_labels)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()
        train_bar = tqdm(train_dataloader)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # print('input_ids', input_ids)
            # print('input_ids shape', input_ids.shape)
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # 验证
        with torch.no_grad():
            model.eval()
            losses = 0  # 损失
            accuracy = 0  # 准确率
            valid_bar = tqdm(valid_dataloader)
            for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
                valid_bar.set_description('Epoch %i valid' % epoch)
                output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    token_type_ids=token_type_ids.to(device),
                )

                loss = criterion(output, label_id.to(device))
                losses += loss.item()

                pred_labels = torch.argmax(output, dim=1)  # 预测出的label
                acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
                accuracy += acc
                valid_bar.set_postfix(loss=loss.item(), acc=acc)

            average_loss = losses / len(valid_dataloader)
            average_acc = accuracy / len(valid_dataloader)

            print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

            if average_acc > best_acc:
                best_acc = average_acc
                torch.save(model.state_dict(), 'models/best_model.pkl')


if __name__ == '__main__':
    main()
