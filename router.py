from flask import Flask, render_template, request


def index():
    return render_template('index.html')
