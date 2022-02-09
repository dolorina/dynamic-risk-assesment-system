import requests
import json
import os

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path =  os.path.join(config['output_model_path']) 



# URL that resolves to workspace
URL = "http://192.168.0.180:8000/" # "http://127.0.0.1:8000"



# Calling each API endpoint and storing the responses
response1 = requests.get(URL+ "prediction?filelocation=./testdata/testdata.csv").content
response2 = requests.get(URL + "scoring").content
response3 = requests.get(URL + "summarystats").content
response4 = requests.get(URL + "diagnostics").content
print(type(response1))
print(response1)
# print(type(response2))
# print(type(response3))
# print(type(response4))

# Combining all API responses
# responses = list([response1, response2, response3, response4])
responses = list([response2, response3, response4])




# # Writing the responses into a file called "apireturns.txt" in the directory specified in output_model_path from config.json
# with open(os.getcwd() + "/" + model_path + "/apireturns.txt", "w") as f:
#     for resp in responses:
#         print(type(resp))
#         f.write(resp)

