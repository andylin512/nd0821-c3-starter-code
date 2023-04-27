"""
Post to live FastAPI instance for model prediction
author: Andy L.
Date: April 26th 2023
"""

import requests, json

SAMPLE =  {  
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

URL = 'https://desolate-coast-82530.herokuapp.com/inference'


# Post rquest with data for prediction 
response = requests.post(url=URL, data = json.dumps(SAMPLE))

print(f"Response status: {response.status_code}")
print(f"Resonse content{response.content}")
print(f"Response JSON OUTPUT {response.json()}")