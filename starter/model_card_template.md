# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest claissfied has been used to predict the salary based on features provided by the census data. All model hyperparametres are default. 
## Intended Use
This is used for exercise only. Not meant to be in real production usage.
## Training Data
80% of the raw data set, after the randomization.
## Evaluation Data
20% of the raw data set, after the randomization. 
## Metrics
3 metrics have been used to evaluated the model performance. They are precision, recall, F1 score. Based on the evaluation data set, the model has:
* precision: 0.71
* recall: 0.63
* fbeta: 0.67
## Ethical Considerations
None
## Caveats and Recommendations
We can try tune the hyperparameters and used other ensemble models, like boosting to see if the model performance can be improved. 