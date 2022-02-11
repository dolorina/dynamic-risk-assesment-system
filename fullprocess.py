'''
Script that fully automates the whole machine learning pipeline: 
- checking for new data
- checking for model drift
- if model drift appeard: retraining and redepoying
Author: Marina Dolokov
Date: February
'''

#!/home/marina/anaconda3/envs/dynamic-risk-assessment/bin/python3.8
from asyncio import subprocess
from doctest import OutputChecker
import pickle
import json
import os
import ast
import pandas as pd
import numpy as np

from ingestion import merge_multiple_dataframe
from diagnostics import model_predictions
from reporting import confusion_matrix
from scoring import score_model
from training import train_model
from deployment import store_model_into_pickle
from apicalls import call_api

# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f) 
prod_dep_path = config["prod_deployment_path"]
input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


# Read all needed files
with open(os.getcwd() + "/" + prod_dep_path + "/latestscore.txt", "r") as f:
    f1_latest = ast.literal_eval(f.read())
with open(os.getcwd() + "/" + prod_dep_path + "/trainedmodel.pkl", "rb") as f:
    model_latest = pickle.load(f) 
with open(os.getcwd() + "/" + prod_dep_path + "/ingestedfiles.txt", "r") as f:
    ingestedfiles = f.read() 
newFilenames = os.listdir(os.getcwd() + "/" + input_folder_path)


# take only most recent data for scores of new predictions
found_new_data = False
for newFile in newFilenames: 
    if ingestedfiles.find(newFile)==-1:
        found_new_data = True
    df = pd.read_csv(os.getcwd() + "/" + input_folder_path + "/" + newFile)
    df = df.append(df)
newData = df.drop_duplicates()

# 1. Decision whether to proceed: proceed, if new data was found. 
if found_new_data==True: 
    merge_multiple_dataframe()
    Data = pd.read_csv(os.getcwd() + "/" + output_folder_path + "/finaldata.csv")
    
    # Data or newData for scoring???
    y_pred, _ = model_predictions(model=model_latest, testdata=Data) # , testdata=newData)
    f1_current = score_model(model=model_latest, testdata=Data) # testdata=newData)
    # raw comparison test
    model_drift_occured = (np.max(f1_latest) > f1_current) # >!!
    print("model drift occured: ", model_drift_occured, f1_latest, f1_current)

    # 2. Decision whether to proceed: proceed, if model drift was found. 
    if model_drift_occured==False: 
        train_model()
        store_model_into_pickle()
        with open(os.getcwd() + "/" + prod_dep_path + "/trainedmodel.pkl", "rb") as f:
            newModel = pickle.load(f) 
        confusion_matrix(newModel)
        call_api() # works only of APP is running