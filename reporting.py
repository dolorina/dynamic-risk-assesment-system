'''
Script that does automated reporting.

Author: Marina Dolokov
Date: February 2022
'''
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions



# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path =  os.path.join(config['output_model_path']) 


# filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
# model = pickle.load(filehandler)
def confusion_matrix(model): # =model):
    '''
    Function for reporting: calculating a confusion matrix using the test data and the deployed model
    '''
    y_pred, y_true = model_predictions(model)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    fig = plt.figure()
    plt.matshow(confusion_matrix)
    plt.title('Confusion matrix - risk assessment')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.getcwd() + "/" + model_path + "/confusionmatrix.png")



if __name__ == '__main__':
    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
    model = pickle.load(filehandler)
    confusion_matrix(model)
