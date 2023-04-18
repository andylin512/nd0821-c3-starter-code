# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest claissfied has been used to predict the salary based on features provided by the census data. All model hyperparametres are default. 
## Intended Use
The mode can be used to predict the salary level based on several provided features from the census data. It is worth to noting that this is meant for research purpose only. 
## Training Data
The Census Income data set was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 80% of the raw data set, after the randomization, was used in the training data set. We performed the whitespace stripping in the raw data before passing it inot the data pipeline. 
## Evaluation Data
20% of the raw data set, after the randomization. 
## Metrics
3 metrics have been used to evaluated the model performance. They are precision, recall, F1 score. Based on the evaluation data set, the model has:
* precision: 0.71
* recall: 0.63
* fbeta: 0.67
## Ethical Considerations
Given the data might not be a well representation of the total salary disctribution. Any inferences from this model/data should be use in cautious to understand the reall salary level for specific population. 
## Caveats and Recommendations
We can try tune the hyperparameters and used other ensemble models, like boosting to see if the model performance can be improved. 