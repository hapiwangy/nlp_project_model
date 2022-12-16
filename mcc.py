import base64
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import io
import datetime
import time
import random
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib


def mcc_score(file):
    matplotlib.use('Agg')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model/model.pth')
    df = pd.read_csv(file, delimiter=',',
                     header=None, names=['sentence', 'Tag'])
    sentences = df.sentence.values
    tags = df.Tag.values
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(tags)

    batch_size = 32
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()
    predictions, true_labels = [], []

    def flat_accuracy(preds, Tag):
        #pred_flat = np.argmax(preds, axis=1).flatten()
        # Tag_flat = Tag.flatten()
        print(np.sum(preds == Tag))
        print(len(Tag))
        return np.sum(preds == Tag) / len(Tag)

    for batch in prediction_dataloader:
        # 將資料載入到 gpu 中
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # 不需要計算梯度
        with torch.no_grad():
            # 前向傳播，獲取預測結果
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 儲存預測結果和 labels
        predictions.append(logits)
        true_labels.append(label_ids)
    matthews_set = []

    # 計算每個 batch 的 MCC
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # 計算該 batch 的 MCC
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)
    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    plt.title('MCC Score per Batch')
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()
    flat_predictions = np.concatenate(predictions, axis=0)

    # 取每個樣本的最大值作為預測值
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # 合併所有的 labels
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # 計算 MCC
    mcc = round(matthews_corrcoef(flat_true_labels, flat_predictions), 3)

    acc = flat_accuracy(flat_predictions, flat_true_labels)
    f1 = f1_score(flat_true_labels, flat_predictions,  average='macro')

    return plot_data, mcc, acc, f1
