import torch
from transformers import BertTokenizer
import numpy as np
from nltk.tokenize import sent_tokenize


def model_predict(file):
    result = []
    with open(file, 'r') as f:
        sentenses = sent_tokenize(f.read())
    device = torch.device("cpu")
    model = torch.load('model/model.pth')
    model.eval()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=True)
    for sentense in sentenses:
        encoded_dict = tokenizer.encode_plus(
            sentense,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        output = model(encoded_dict['input_ids'], token_type_ids=None)
        logits = output[0]
        logits = logits.detach().cpu().numpy()
        flat_prediction = np.argmax(logits, axis=1).flatten()
        with open('./classes.txt', 'r') as f:
            result.append(
                (sentense, f.readlines()[flat_prediction[0]].replace('\n', '')))
    return result
