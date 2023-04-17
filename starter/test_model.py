"""
Unit test of model module with pytest
author: Andy L.
Date: Apirl 16th 2022
"""

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import train_model
from starter.ml.data import process_data

# Create Fixture
@pytest.fixture(scope="module")
def path():
    return "./data/census.csv"

@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("./data/census.csv")
    return df

@pytest.fixture(scope="module")
def feature():
    features = [    
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"
                    ]
    return features

@pytest.fixture(scope="module")
def train_dataset(data, feature):
    train, test = train_test_split( 
                                data, 
                                test_size=0.20, 
                                random_state=123
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=feature,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train

# Test 
def test_load_data(path):
    """
    Test if the path exsits valid data frame
    """
    df = pd.read_csv(path)
    assert df.shape[0]>0
    assert df.shape[1]>0

def test_features(data, feature):
    """
    Test if loaded data set have matched category data
    """
    assert all(feat in data.columns for feat in feature)

def test_modelling(train_dataset):
    """
    Test if a valid random forest classifiers returned 
    """
    X_train, y_train = train_dataset
    output_model = train_model(X_train, y_train)
    assert isinstance(output_model, RandomForestClassifier)