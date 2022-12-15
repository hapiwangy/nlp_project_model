from flask import Flask, render_template
from router import index

app = Flask(__name__)


app.add_url_rule('/index', 'index',
                 index, methods=['GET', 'POST'])

app.add_url_rule('/', 'index',
                 index, methods=['GET', 'POST'])


if __name__ == '__main__':
    app.debug = True
    app.run()
