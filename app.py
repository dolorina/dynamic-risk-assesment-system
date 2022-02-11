from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary, count_missing_data, execution_time, outdated_packages_list
from scoring import score_model


with open('config.json','r') as f:
    config = json.load(f) 

model_path =  os.path.join(config['output_model_path']) 
prod_dep_path =  os.path.join(config['prod_deployment_path']) 

filehandler = open(os.getcwd() + "/" + prod_dep_path + "/trainedmodel.pkl", "rb")
model = pickle.load(filehandler)

# Set up variables for use in script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


# Prediction Endpoint
@app.route("/prediction", methods=['GET'])
def predict():   
    filelocation = request.args.get('filelocation')
    dataset = pd.read_csv(filelocation)   
    y_pred, _ = model_predictions(model=model, testdata=dataset)
    string_ypred = str(y_pred)
    return string_ypred


# Scoring Endpoint
@app.route("/scoring", methods=['GET'])
def score(): 
    f1_score = str(score_model(model=model))
    return f1_score


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET'])
def summarize():        
    statistics = str(dataframe_summary())
    return statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET'])
def diagnose():        
    NA_percent = str(count_missing_data()) 
    time = str(execution_time())
    outdated = str(outdated_packages_list())
    return "NA in %: " + NA_percent + "\n" + "Execution time: " + time + "\n" + outdated 


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)