
import os
import json
import pandas as pd
import fastavro
import io
import cloudpickle

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Required by SageMaker for loading the model
def model_fn(model_dir):
    """
    Loads the trained TimeSeriesPredictor from the model directory.
    
    Args:
        model_dir (str): The directory where the model artifact is located.
        
    Returns:
        TimeSeriesPredictor: The loaded predictor.
    """
    return TimeSeriesPredictor.load(model_dir)

# Required by SageMaker for inference
def transform_fn(predictor, data, content_type, accept_type):
    """
    Handles data deserialization, prediction, and serialization.
    
    Args:
        predictor (TimeSeriesPredictor): The loaded Autogluon predictor.
        data (str or bytes): The input data.
        content_type (str): The Content-Type header of the input data.
        accept_type (str): The Accept header requested by the client.
        
    Returns:
        bytes: The serialized prediction result.
        str: The Content-Type of the prediction result.
    """
    # 1. Deserialize input
    if content_type == "application/json":
        # Handle pandas-split format from JSON
        df = pd.read_json(io.StringIO(data), orient="split")
    elif content_type == "application/x-avro-bytes":
        # Handle Avro bytes from a custom content type
        input_stream = io.BytesIO(data)
        reader = fastavro.reader(input_stream)
        records = [r for r in reader]
        df = pd.DataFrame.from_records(records)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    # 2. Perform prediction
    ts_dataframe = TimeSeriesDataFrame(df)
    predictions = predictor.predict(ts_dataframe)

    # 3. Serialize output
    if accept_type == "application/json":
        return predictions.to_json(orient="split"), accept_type
    elif accept_type == "text/csv":
        return predictions.to_csv(), accept_type
    else:
        raise ValueError(f"Unsupported accept type: {accept_type}")
