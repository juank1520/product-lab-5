from flask import Flask, request, jsonify
import pipeline_predict as pp
import json
import pandas as pd


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, world</p>"

@app.route("/predict", methods=['POST'])
def predictFromPp():
    json_data = request.get_json()
    json_data = json.dumps(json_data)
    json_data = json.loads(json_data)

    dataFrame = pd.DataFrame.from_dict(json_data, orient='index')
    resultado = pp.predict(dataFrame)
    print(dataFrame)
    return jsonify({'Prediccion': resultado})
    