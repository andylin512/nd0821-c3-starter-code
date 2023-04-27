# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from slice_performance import slice_performance
import json
# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Save encoder, lb for future ref
pd.to_pickle(encoder, "../model/encoder.pkl")
pd.to_pickle(lb, "../model/lb.pkl")

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features, 
    label="salary", 
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
pd.to_pickle(model, "../model/model.pkl")

pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)
with open("slice_output.txt", "w") as f:
    f.write("Overall model performance on test data set:\n")
    json.dump(
        {"precision": precision,
         "recall": recall,
         "fbeta": fbeta},
         f,
         indent=6
    )
    f.write("\n")

# Save slice performance output: take education as an example
edu_slice_perf, feat = slice_performance(model,test, cat_features, "education", encoder, lb)
with open("slice_output.txt", "a") as f:
    f.write(f"{feat}-feature slices model performance on test data set:\n")
    json.dump(
       edu_slice_perf,
       f,
       indent=6
    )
    f.write("\n")

