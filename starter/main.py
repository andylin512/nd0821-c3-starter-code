# Put the code for your API here.
"""
FastAPI to host the model
author: Andy L.
Date: April 17 2023

"""
from fastapi import FastAPI, HTTPException
from typing import Union, Optional
from pydantic import BaseModel
import pandas as pd
import os, pickle, sys
from starter.ml.data import process_data

# Define Data object and type
class Data(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str




# Init. FastAPI app
app = FastAPI(
    title="Inference API",
    description="An API that takes a data sample and run the prediction",
    version="1.0.0"
)

@app.get("/")
async def greetings():
    return "Welcome to my first model API"

# Upload sample data and returnt the prediction
@app.post("/inference/")
async def upload_data(inference: Data):
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }
    
    df = pd.DataFrame(data, index=[0])

    # File Dir
    FILE_DIR = "./starter/"# os.path.dirname(os.path.abspath(sys.argv[1]))

    # cat_feature for data transformation purpose
    CAT_FEATURE = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]
    
    # Load pre-trained model and token
    model = pickle.load(open(os.path.join(FILE_DIR,"model/model.pkl"), "rb"))
    encoder = pickle.load(open(os.path.join(FILE_DIR,"model/encoder.pkl"), "rb"))
    lb = pickle.load(open(os.path.join(FILE_DIR,"model/lb.pkl"), "rb"))

    sample, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURE,
        training=False,
        encoder=encoder,
        lb=lb
    )
    pred = model.predict(sample)
    if pred[0] > 0.5:
        pred_lb = '>50k'
    else:
        pred_lb = '<=50k'
    df["prediction"] = pred_lb
    return df.to_json(orient="columns")

if __name__ == '__main__':
    pass
