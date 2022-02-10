'''
Unit tests for API App

Author: Marina Dolokov
Date: February 2022 

'''
import requests
import os


# URL that resolves to workspace
URL = "http://192.168.0.180:8000/"


def test_predict():
    r = requests.get(URL+ "prediction?filelocation=" + os.getcwd() + "/testdata/testdata.csv")
    assert r.status_code == 200


def test_score():
    r = requests.get(URL + "scoring")
    assert r.status_code == 200


def test_summarize():
    r = requests.get(URL + "summarystats")
    assert r.status_code == 200


def test_diagnose():
    r = requests.get(URL + "diagnostics")
    assert r.status_code == 200