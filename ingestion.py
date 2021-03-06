'''
Script that does data ingestion 

Author: Marina Dolokov
Date: February 2022
'''
import pandas as pd
import numpy as np
import ast
import os
import json
from datetime import datetime



# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f) 



input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]



def record_ingested_data(allRecords):
    '''
    Function that saves the recording of the ingested data to output folder
    Args:
        allRecords: numpy array
    '''
    filesinDir = os.listdir(os.getcwd()+ "/" + output_folder_path)
    ingestedfiles_exist = any("ingestedfiles.txt" == file for file in filesinDir)
    # if no existing ingestedfiles.txt file found, create one 
    if ingestedfiles_exist == False: 
        with open(os.getcwd()+ "/" + output_folder_path + "/" + "ingestedfiles.txt","w") as f:
            for rec in allRecords:
                f.write(str(rec)+"\n")
    # if found, append new data to it
    else:    
        for file in filesinDir: # only look for ingestedfiles.txt, not for other files in directory
            # make sure not to double save the file names 
            not_existing_filename = any(file!= rec[1] for rec in allRecords)
            if file.find("ingestedfiles.txt")!=-1 and not_existing_filename:
                with open(os.getcwd()+ "/" + output_folder_path + "/" + "ingestedfiles.txt","r") as f:
                    oldRecords = f.read()
                with open(os.getcwd()+ "/" + output_folder_path + "/" + "ingestedfiles.txt","w") as f:    
                    f.write(oldRecords + "\n")
                    for rec in allRecords:
                        f.write(str(rec) + "\n")



def merge_multiple_dataframe(new_data_found=True):
    '''
    Function for data ingestion: checks for datasets, complies them together and writes them to an output file
    '''
    # check for files in input_folder_path 
    filenames = os.listdir(os.getcwd()+ "/" + input_folder_path)
    # create a pandas dataframe for csv files that will be read 
    df_list = pd.DataFrame(columns=["corporation", "lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"])
    # create numpy array for keeping record of ingested data
    allRecords = np.array([])


    # check for datasets and amerge them together in a pandas dataframe
    for file in filenames: 
        df1 = pd.read_csv(os.getcwd()+ "/" + input_folder_path + "/" + file)
        df_list = df_list.append(df1)
        
        # keeping record of merged datasets
        dateTimeObj = datetime.now()
        thetimenow=str(dateTimeObj.year)+ '/' +str(dateTimeObj.month)+ '/' +str(dateTimeObj.day)
        record=[input_folder_path, file, len(df1.index),thetimenow]
        allRecords = np.append(allRecords, record)
    allRecords = np.reshape(allRecords, (-1, len(record)))
    
    record_ingested_data(allRecords)

    # drop duplicates in merged dataframe and write the final dataframe to an output csv file   
    df_result = df_list.drop_duplicates()
    df_result.to_csv(output_folder_path + "/finaldata.csv", index=False)

    


if __name__ == '__main__':
    merge_multiple_dataframe()
