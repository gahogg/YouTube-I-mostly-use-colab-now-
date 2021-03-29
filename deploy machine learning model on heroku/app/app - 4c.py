import numpy as np
from joblib import dump, load
from flask import render_template, request
from flask import Flask

app = Flask(__name__)
loaded_model = load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('main.html')
    else:
        text = request.form['text']
        processed_text = text.upper()
        return processed_text