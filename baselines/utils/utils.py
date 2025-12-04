import numpy as np
from sklearn.metrics import homogeneity_score

def data_preprocessing(data):
    """
    Preprocess the data. Convert the data into the format required by MixtureSlicer.
    """
    # Assuming data is a DataFrame with columns 'embeddings', 'targets', and 'pred_probs'
    # Convert the DataFrame to numpy arrays
    processed_data = {}
    for col in data.columns:
        processed_data[col] = np.array(data[col].values.tolist())     
    return processed_data

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')
