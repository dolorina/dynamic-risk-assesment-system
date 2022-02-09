'''
Script that does diagnosis to help find problems (if any exist).

Author: Marina Dolokov
Date: February 2022
'''
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import ast
import json
import subprocess

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path =  os.path.join(config['output_model_path']) 



def model_predictions():
    '''
    Funciton that return predictions made by the deployed model
    Args: 
        testdata (pandas DataFrame) data for model predictions 
    Returns:
        y_pred (list) list with predictions
    '''
    testdata = pd.read_csv(os.getcwd() + "/" + test_data_path + "/testdata.csv")
    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
    model = pickle.load(filehandler)
    X_test = testdata.drop(["corporation", "exited"], axis=1)
    X_test = X_test.values.reshape(-1, 3)
    # y_test = testdata["exited"].values.reshape(-1, 1).ravel()

    filehandler = open(os.getcwd() + "/" + model_path + "/trainedmodel.pkl", "rb")
    model = pickle.load(filehandler)

    y_pred = list(model.predict(X_test))
    assert len(y_pred)==len(testdata)
    return y_pred 



# Function to get summary statistics
def dataframe_summary():
    '''
    Function to get summary statistics
    Returns: 
        statistics (list) a list containing the mean, median & standard deviation of the numeric columns 
    '''
    data = pd.read_csv(os.getcwd() + "/" + dataset_csv_path + "/finaldata.csv")
    #long version for avoiding errors in future (because of current "FutureWarning")
    means1 = data["lastmonth_activity"].mean()
    means2 = data["lastyear_activity"].mean()
    means3 = data["number_of_employees"].mean()
    means4 = data["exited"].mean()
    medians1 = data["lastmonth_activity"].median()
    medians2 = data["lastyear_activity"].median()
    medians3 = data["number_of_employees"].median()
    medians4 = data["exited"].mean()
    deviations1 = data["lastmonth_activity"].std()
    deviations2 = data["lastyear_activity"].std()
    deviations3 = data["number_of_employees"].std()
    deviations4 = data["exited"].mean()
    statistics = list([[means1, means2, means3, means4], 
                      [medians1, medians2, medians3, medians4], 
                      [deviations1, deviations2,deviations3, deviations4]])
    # # shorter version but with "FutureWarning"
    # means = list(data.mean())
    # medians = list(data.median())
    # deviations = list(data.std())
    # statistics = list([means, medians, deviations])
    return statistics 



def count_missing_data():
    '''
    Function that calculates NA values in datat and return percentages of NA in datat columns
    Returns: 
        NA_percent (list) percentage for every columns of NA values in data
    '''
    data = pd.read_csv(os.getcwd() + "/" + dataset_csv_path + "/finaldata.csv")
    NAs = list(data.isna().sum())
    NA_percent = [NAs[i]/len(data.index) for i in range(len(NAs))]
    return NA_percent



def execution_time():
    '''
    Function that measures timing of ingestion.py and training.py in seconds
    Returns: 
        time (list) a list containing the running time of ingestion.py and training.py
    '''
    starttime1 = timeit.default_timer()
    os.system("python training.py")
    timing1=timeit.default_timer() - starttime1

    starttime2 = timeit.default_timer()
    os.system("python training.py")
    timing2=timeit.default_timer() - starttime2

    time = list([timing1, timing2])
    return time



def outdated_packages_list():
    '''
    Function to check dependencies. Functions saves dependencies into a list in the file outdated_packages.txt
    '''
    installed = subprocess.check_output(['pip', 'list', '--outdated'])
    with open('outdated_packages.txt', 'wb') as f:
        f.write(installed)
    # with open('outdated_packages.txt', 'r') as f:
    #     outdated_packages = f.read()
    # print(outdated_packages) 

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()