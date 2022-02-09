'''
Script that trains a logistic regression on provided data

Author: Marina Dolokov
Date: February 2022
'''

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


# Function for training the model
def train_model():
    
    # Reading in finaldata.csv using pandas 
    data = pd.read_csv(os.getcwd() + "/" + dataset_csv_path + "/finaldata.csv")
    X = data.drop(["corporation", "exited"], axis=1)
    X = X.values.reshape(-1, 3)
    y = data["exited"].values.reshape(-1, 1).ravel()

    # Logistic regression for training
    lg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # Fitting logistic regression to provided datagit
    model = lg.fit(X, y)

    # Writing trained model to model_path in the file trainedmodel.pkl
    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "wb")
    pickle.dump(model, filehandler)


if __name__ == '__main__':
    train_model()