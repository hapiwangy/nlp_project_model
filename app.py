from flask import Flask, render_template
import torch
import transformers
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

from router import index, predict, show, member, process, model_predict
app = Flask(__name__)


app.add_url_rule('/index', 'index',
                 index, methods=['GET', 'POST'])

app.add_url_rule('/', 'show',
                 show, methods=['GET', 'POST'])
app.add_url_rule('/show', 'show',
                 show, methods=['GET', 'POST'])
app.add_url_rule('/member', 'member',
                 member, methods=['GET', 'POST'])
app.add_url_rule('/process', 'process',
                 process, methods=['GET', 'POST'])
app.add_url_rule('/model_predict', 'model_predict',
                 model_predict, methods=['POST'])

app.add_url_rule('/predict', 'predict',
                 predict, methods=['GET', 'POST'])


if __name__ == '__main__':
    app.debug = True
    app.run()
