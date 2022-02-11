import requests
import json
import os

from reporting import create_reporting_pdf

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 
model_path =  os.path.join(config['output_model_path']) 
test_data_path =  os.path.join(config['test_data_path'])


def call_api():
    # URL that resolves to workspace
    # try:
    URL = "http://192.168.0.181:8000"
    # except:
    #     URL = "http://192.168.0.181:8000"
        
    # Calling each API endpoint and storing the responses
    response1 = requests.get(URL+ "/prediction?filelocation=" + os.getcwd() + "/" + test_data_path + "/testdata.csv")
    response2 = requests.get(URL + "/scoring")
    response3 = requests.get(URL + "/summarystats")
    response4 = requests.get(URL + "/diagnostics")
    
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

    create_reporting_pdf()



if __name__ == '__main__':
    call_api()