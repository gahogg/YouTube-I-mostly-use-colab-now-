import numpy as np
from joblib import dump, load
from flask import Flask

app = Flask(__name__)
loaded_model = load('model.joblib')

@app.route('/')
def hello_world():
    test_x = np.array([[1], [2]])
    preds = loaded_model.predict(test_x)
    output_str = ''

    for inp, outp in zip(test_x, preds):
        output_str += 'Input: ' + str(inp) + ' ==> Output: ' + str(outp) + ' '
    
    return output_str