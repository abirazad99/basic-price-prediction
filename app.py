# import pickle 
# we have a file named pickle.py in our working directory, 
# it will be imported instead of the standard library module pickle. 
# This can lead to unexpected behavior and errors, 
# as the contents of our pickle.py file may not be compatible with the standard library's pickle module.

# import pickle as stdlib_pickle (doesn't work)


# This will import the standard library pickle module 
# and assign it to the variable pickle, 
# even if we have a file named pickle.py in our working directory.
import importlib
pickle = importlib.import_module("pickle")

from flask  import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas  as pd

app = Flask(__name__)
## Loading the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  # this srender_template will look for a templates folder

@app.route('/predict_api', methods = ['POST'])  # we are usin /predict_api as an api itself with POST
def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1, -1))
    new_scaled_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_scaled_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run (debug = True)

