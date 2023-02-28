from flask import Flask, render_template
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
