import statistics
import string
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os

from diagnostics import model_predictions, dataframe_summary, count_missing_data, execution_time, outdated_packages_list
from scoring import score_model



with open('config.json','r') as f:
    config = json.load(f) 

model_path =  os.path.join(config['output_model_path']) 


# prediction_model = None

# Set up variables for use in script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'



# Prediction Endpoint
@app.route("/prediction/", methods=['POST','OPTIONS'])
def predict():   
    filelocation = request.args.get('filelocation')     
    dataset = filelocation # + "/finaldata.csv"
    y_pred, _ = model_predictions(dataset)
    return y_pred 



# Scoring Endpoint
@app.route("/scoring/", methods=['GET','OPTIONS'])
def score(): 
    f1_score = score_model()       
    return f1_score 



# Summary Statistics Endpoint
@app.route("/summarystats/", methods=['GET','OPTIONS'])
def summarize():        
    statistics = dataframe_summary()
    return statistics



# Diagnostics Endpoint
@app.route("/diagnostics/", methods=['GET','OPTIONS'])
def diagnose():        
    NA_percent = count_missing_data() 
    time = execution_time()
    outdated = outdated_packages_list()
    return NA_percent, time, outdated 



if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
