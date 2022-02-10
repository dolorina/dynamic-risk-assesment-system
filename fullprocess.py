
import json
import os

from ingestion import merge_multiple_dataframe
# import training
# import scoring
# import deployment
# import diagnostics
# import reporting


# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f) 

prod_dep_path = config["prod_deployment_path"]
input_folder_path = config["input_folder_path"]

# Check and read new data
with open(os.getcwd() + "/" + prod_dep_path + "/ingestedfiles.txt", "r") as f:
    ingestedfiles = f.read() 

datafiles = os.listdir(os.getcwd() + "/" + input_folder_path)

found_new_data = False
for file in datafiles: 
    if ingestedfiles.find(file)==-1:
        found_new_data = True


if found_new_data==True: 
    merge_multiple_dataframe()


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







