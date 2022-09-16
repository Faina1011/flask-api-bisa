import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neighbors._base
import sys
from tensorflow.keras.models import Sequential, model_from_json
from keras import models    
from keras.models import load_model
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neighbors._base
import sys
from tensorflow.keras.models import Sequential, model_from_json
from keras import models    
from tensorflow.keras.models import load_model

import os
import json
import requests
from json import loads


sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base



from jcopml.utils import load_model
#from model import predict
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
json_file = open('model/modelaus.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/modelaus.h5")
print("Loaded model from disk")

@app.route('/echo', methods=['POST'])
def hello():
    #print(request.get_json())
    #data = json.loads(request.data)
    #content = request.json

    response = request.get_json()
    
    json_string = json.dumps(response)
    

    X_test = pd.read_json(json_string, orient="rain")
    X_test["rain tomorrow"] = loaded_model.predict(X_test)
    X_test["rain tomorrow"] = (X_test["rain tomorrow"] > 0.5)

    data = X_test.to_json()
    
    return (data)



if __name__ == "__main__":
    app.run(debug=True, port=5005)

