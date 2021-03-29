import numpy as np
from joblib import dump, load
from flask import render_template
from flask import Flask

app = Flask(__name__)
loaded_model = load('model.joblib')

@app.route('/')
def hello_world():
    return render_template('main.html')