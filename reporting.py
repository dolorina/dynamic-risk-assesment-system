'''
Script that does automated reporting.

Author: Marina Dolokov
Date: February 2022
'''
import pickle
import ast
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from reportlab.pdfgen import canvas

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


def create_reporting_pdf():
    from reportlab.lib.pagesizes import letter

    with open(os.getcwd() + "/" + model_path + "/latestscore.txt", "r") as f:
        f1_score = f.read()
    
    with open(os.getcwd() + "/" + model_path + "/apireturns.txt", "r") as f:
        api_returns = f.readlines()


    pdf = canvas.Canvas('report.pdf', pagesize=letter)
    width, height = letter
    pdf.drawString(30,height - 100,'Dynamic risk assessment - Report')
    pdf.drawString(30,height - 115,'API returns: ')
    i = 130
    for a in api_returns:
        pdf.drawString(30,height - i, a)
        i +=15
    
    pdf.drawString(30,height - (i+15),'F1 score of latest model: '+ f1_score)
    pdf.drawInlineImage(os.getcwd() + "/" + model_path + "/confusionmatrix.png", 30, height-(i+30)-300, 350, 300)
    pdf.save()


if __name__ == '__main__':
    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
    model = pickle.load(filehandler)
    confusion_matrix(model)