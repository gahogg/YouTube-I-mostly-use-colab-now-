import numpy as np
import pandas as pd
from joblib import dump, load
from flask import url_for, render_template, request, send_from_directory, make_response
from flask import Flask
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly
from os.path import dirname, join
import uuid

app = Flask(__name__)
loaded_model = load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('main.html', href=url_for('static', filename="base_pic.svg"))
    else:
        text = request.form['text']
        np_arr = floats_string_to_input_arr(text)
        u = uuid.uuid4()
        random_url = join("static/", u.hex + ".svg")
        make_picture('AgesAndHeights.pkl', np_arr, loaded_model, output_file=random_url)
        return render_template('main.html', href=random_url)

# After Step 3
def make_picture(training_data_filename, new_input_arr, model, output_file):
  # Plot training data with model
  data = pd.read_pickle('AgesAndHeights.pkl')
  x_new = np.arange(18).reshape((18, 1))
  preds  = model.predict(x_new)
  ages = data['Age']
  heights = data['Height']
  fig = px.scatter(x=ages, y=heights, title="Height vs Age", labels={'x': 'Age (Years)',
                                                                   'y': 'Height (Inches)'})
  fig.add_trace(
      go.Scatter(x=x_new.reshape(x_new.shape[0]), y=preds, mode='lines', name='Model'))

  if new_input_arr is not False:
    # Plot new predictions
    new_preds = model.predict(new_input_arr)
    fig.add_trace(
      go.Scatter(x=new_input_arr.reshape(new_input_arr.shape[0]), y=new_preds, name='New Outputs', mode='markers', marker=dict(
            color='purple',
            size=20,
            line=dict(
                color='purple',
                width=2
            ))))
  fig.write_image(output_file, width=800, engine='kaleido')
  return fig

def floats_string_to_input_arr(floats_str):
  floats = [float(x.strip()) for x in floats_str.split(',')]
  as_np_arr = np.array(floats).reshape(len(floats), 1)
  return as_np_arr