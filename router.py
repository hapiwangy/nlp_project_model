from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from mcc import mcc_score


def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            file.save('data/'+file.filename)
            mcc_score('data/'+file.filename)
    return render_template('index.html')
