import torch
import transformers
from transformers import BertTokenizer
import numpy as np
import sys
labels = ["not included","Climate","Natural capital","Pollution&Waste","Env. Opportunities","Human Capital","Product Liability","Social Opportunities","Corporator Governance","Corporator Behavior"]
device = 'cpu' 
# 使用cpu裝置
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', outputs_attentions=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 9)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
# 設定自訂模型架構(ABC)
model = torch.load(r"你的模型路徑", map_location='cpu')
# 載入自訂模型參數(要記得換成裡的模型路徑)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
MAX_LEN = 256
# 設定基本數值
inputs = sys.argv[1]
# 獲得輸入(等等把它改成cmd輸入)
after_encode_inputs = tokenizer.encode_plus(
    inputs,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding=True,
    return_token_type_ids=True,
    truncation=True
)
# 進行encode
ids = torch.tensor([after_encode_inputs['input_ids']], dtype=torch.long)
mask = torch.tensor([after_encode_inputs['attention_mask']], dtype=torch.long)
token_type_ids = torch.tensor([after_encode_inputs["token_type_ids"]], dtype=torch.long)
# 取出需要的資料
model.eval()
with torch.no_grad():
    outputs = model(ids, mask, token_type_ids)
outputs = np.array(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
outputs = outputs >= 0.4
outputs = outputs[0]
ans = []
for index , x in enumerate(outputs):
    if x == True:
        ans.append(labels[index])
ans = ",".join(ans)
if len(ans) == 0:
    print(f"this sentence is unrelated!!")
else:
    print(f"this sentence's labels include {ans}")
# 進行判斷並輸出結果