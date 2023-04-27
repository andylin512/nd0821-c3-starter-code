
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def slice_performance(model,data, cat_features, col, encoder, lb):
    """This function slices performance metrics form `compute_model_metrics` by different
    feature values (col) and output the dict

    Inputs
    ------
    model:???
        Trained machine learning model.
    data: pandas dataframe
        The dataframe that is used for validation purpose
    cat_features: list
        The list of the categorical feature names
    col: str
        The name of the column that we want to slice the model performance
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    slice_perf: dict
        model performance metrics, wrapped in a dictionary
    feat: str
        name of the given features
    """
    cats = data.loc[:, col].unique()
    slice_perf = {}
    for Slice in cats:
        data_slice = data.loc[data.loc[:, col]==Slice,:]
        X_slice, y_slice, _, _ = process_data(
                                        data_slice,
                                        categorical_features=cat_features, 
                                        label="salary", 
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)
        pred_slice = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, pred_slice)
        
        slice_perf[f"{col}_{Slice}"]={}
        slice_perf[f"{col}_{Slice}"]["precision"] = precision
        slice_perf[f"{col}_{Slice}"]["recall"] = recall
        slice_perf[f"{col}_{Slice}"]["fbeta"] = fbeta
    return slice_perf, col
