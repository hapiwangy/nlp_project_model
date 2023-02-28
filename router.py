from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from mcc import mcc_score
from model_predict import model_predict


def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            file.save('data/'+file.filename)
            mcc_url, mcc, acc, f1 = mcc_score('data/'+file.filename)
            return render_template('index.html', mcc_url=mcc_url, mcc=mcc, acc=acc, f1=f1)
    return render_template('index.html')


def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            file.save('data/'+file.filename)
            return render_template('predict.html', results=model_predict('data/'+file.filename))
    return render_template('predict.html', results='')


def show():
    try:
        return render_template('show.html', result=request.args['result'])
    except:
        return render_template('show.html', result='')


def process():
    return render_template('process.html')


def member():
    return render_template('member.html')


def model_predict():
    return redirect(url_for('show', result=request.form['text']))
