import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'

label_dic = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']


def main():

    # 参数设置
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 2
    learning_rate = 5e-6  # Learning Rate不宜太大
    model_name_or_path = 'chinese_roberta_wwm_ext_pytorch'
    model_name_or_path = 'chinese_xlnet_mid_pytorch'

    df = pd.read_table('./data/cnews/cnews.train.txt', header=None)
    df.columns = ['label', 'content']

    # 获取到dataset
    train_dataset = CNewsDataset('data/cnews/cnews.train.txt', model_name_or_path)
    valid_dataset = CNewsDataset('data/cnews/cnews.val.txt', model_name_or_path)
    # test_data = load_data('cnews/cnews.test.txt')

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_name_or_path)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(1, epochs + 1):
        losses = 0      # 损失
        accuracy = 0    # 准确率

        model.train()
        train_bar = tqdm(train_dataloader)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
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
            losses = 0      # 损失
            accuracy = 0    # 准确率
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

                pred_labels = torch.argmax(output, dim=1)   # 预测出的label
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
