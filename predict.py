import torch
from model import BertClassifier
from transformers import BertTokenizer, BertConfig


labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
bert_config = BertConfig.from_pretrained('bert-base-chinese')

model = BertClassifier(bert_config, len(labels))
model.load_state_dict(torch.load('models/best_model.pkl', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

print('新闻类别分类')
while True:
    text = input('Input: ')
    token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    token_type_ids = token['token_type_ids']

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    predicted = model(
        input_ids,
        attention_mask,
        token_type_ids,
    )
    pred_label = torch.argmax(predicted, dim=1)

    print('Label:', labels[pred_label])
