'''
Script that scores a logistic regression model on provided test data

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
test_data_path = os.path.join(config['test_data_path'])
model_path =  os.path.join(config['output_model_path'])



def save_score(f1_score):
    '''
    Function that saves the achieved model score
    Args:
        f1_score: double
    '''
    with open(os.getcwd() + "/" + model_path + "/latestscore.txt", "w") as f:
        f.write(str(f1_score))



def score_model():
    '''
    This function takes a trained model, loads test data and calculates a F1 score for the model relative to the test data.
    It writes the result to the latestscore.txt file    
    '''

    testdata = pd.read_csv(os.getcwd() + "/" + test_data_path + "/testdata.csv")
    X_test = testdata.drop(["corporation", "exited"], axis=1)
    X_test = X_test.values.reshape(-1, 3)
    y_test = testdata["exited"].values.reshape(-1, 1).ravel()

    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
    model = pickle.load(filehandler)

    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_pred, y_test)

    save_score(f1_score)
    return f1_score

    
if __name__ == '__main__':
    score_model()