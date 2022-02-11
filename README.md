# dynamic-risk-assesment-system

[Link to this GitHub Repository](https://github.com/dolorina/dynamic-risk-assesment-system)

In this project a risk assessment ML model is created, deployed and monitored. 

## Possible Setting for model use 
A company has 10,000 corporate clients. This company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If this model is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

But with creating and deploying the model the work does not end, though. The companies industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, a regular monitoring of the model most be set up to ensure that it remains accurate and up-to-date. 

In this project processes and scripts for retraining, redeployment, monitoring and reporting on the ML model are set up, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project Steps Overview

1. **Data ingestion:** 
* Automatically checks a database for new data that can be used for model training
* Compiles all training data to a training dataset and saves it to persistent storage
* Writes metrics related to the completed data ingestion tasks to persistent storage.
    
2. **Training, scoring, and deploying:**
* Writes scripts that train an ML model that predicts attrition risk, and score the model. 
* Writes the model and the scoring metrics to persistent storage.


3. **Diagnostics:** 
* Determines and saves summary statistics related to a dataset.
* Times the performance of model training and scoring scripts.
* Checks for dependency changes and package updates.
    
4. **Reporting:** 
* Automatically generates plots and documents that report on model metrics. 
* Provides an API endpoint that can return model predictions and metrics.

5. **Process Automation:** 
* Creates a script and cron job that automatically run all previous steps at regular intervals.


## Run project

1. If you run the project for the first time and have no trained model saved yet, you have to use the folders specified in config_for_first_run.json

2. If you want to run the full process, run the following line:

```
$ python app.py
$ python fullprocess.py
```

3. As alternative you can set up a cronjob using the lines which are saved in cronjob.txt. There fore type in terminal

```
$ service cron start
$ crontab -e 
```

Copy the lines from cronjob.txt and paste them into the cronfile. Run the following line, to check the cronjob:

```
$ crontab -l 
```