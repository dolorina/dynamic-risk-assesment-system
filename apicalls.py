import requests
import json
import os

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 
model_path =  os.path.join(config['output_model_path']) 

def call_api():
    # URL that resolves to workspace
    URL = "http://192.168.0.180:8000/" # "http://127.0.0.1:8000"
    # Calling each API endpoint and storing the responses
    response1 = requests.get(URL+ "prediction?filelocation=" + os.getcwd() + "/testdata/testdata.csv")
    response2 = requests.get(URL + "scoring")
    response3 = requests.get(URL + "summarystats")
    response4 = requests.get(URL + "diagnostics")


    # Combining all API responses
    responses = list([
        "Predictions: " + response1.text+"\n", 
        "\n", 
        "F1 score: " + response2.text+"\n", 
        "\n", 
        "Statistics: \n" + response3.text+"\n", 
        "\n", 
        "Diagnostics: \n" + response4.text])


    with open(os.getcwd() + "/" + model_path + "/apireturns.txt", "w") as f:
        for resp in responses:
            f.write(resp)

