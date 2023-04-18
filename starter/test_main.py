"""
Unit test(pytest) for main.py
author: Andy L.
Date: April 17th 2023
"""

from fastapi.testclient import TestClient
import json
import logging
from main import app
import pytest

CLIENT = TestClient(app)


# Define fixture
@pytest.fixture(scope="module")
def sample_1():
    sample =  {  
            'age':10,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
        }
    return sample

@pytest.fixture(scope="module")
def sample_2():
    sample =  {  
            'age':60,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
        }
    return sample



def test_get():
    """
    Test Get response and message
    """
    r = CLIENT.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to my first model API"

def test_prediction_1(sample_1):
    """
    Test API response when uploading proper sample data
    """
    data = json.dumps(sample_1)
    r = CLIENT.post("/inference/", data=data)
    result = json.loads(r.json())
    
    assert r.status_code == 200
    assert result["age"]["0"]  == 10
    assert result["prediction"]["0"] == "<=50k"

    



def test_prediction_2(sample_2):
    """
    Test prediction result after uploading proper sample data
    """
    data = json.dumps(sample_2)
    r = CLIENT.post("/inference/", data=data)
    result = json.loads(r.json())
    print(result)
    assert result["age"]["0"]  == 60
    assert result["prediction"]["0"] == ">50k"