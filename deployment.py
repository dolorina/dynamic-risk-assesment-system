'''
Script that copies the trained model, the latest score and the record file for the ingested data into production environment 

Author: Marina Dolokov
Date: February 2022
'''
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

ingesteddata_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path =  os.path.join(config['output_model_path']) 

# filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
# model = pickle.load(filehandler)

# Function for deployment
def store_model_into_pickle():
    '''
    Function for deployment
    Args: 
        model (Logistic Regression model from scikit learn)
    '''
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    shutil.copy(model_path + "/trainedmodel.pkl", prod_deployment_path)
    shutil.copy(model_path + "/latestscore.txt", prod_deployment_path)
    shutil.copy(ingesteddata_path + "/ingestedfiles.txt", prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()