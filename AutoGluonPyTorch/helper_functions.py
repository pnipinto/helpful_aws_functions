import json
import pandas as pd
import mlflow
import tempfile
import tarfile
import fastavro
import io
import os
import cloudpickle
import time
from functools import wraps
import pyarrow.parquet as pq

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from mlflow.pyfunc import PythonModel, log_model
from mlflow.tracking import MlflowClient
import sagemaker


# ----------------------------
# Retry helper
# ----------------------------
# A decorator to automatically retry a function call on failure with exponential backoff.
# This is useful for dealing with transient errors in file systems or network calls.
def retry_decorator(max_attempts=3, delay_seconds=60, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts, delay = 0, delay_seconds
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    print(f"[retry] {e} | attempt {attempts}/{max_attempts} | sleeping {delay}s")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


class AGTimeSeriesWrapper(PythonModel):
    """
    A custom MLflow Pyfunc model wrapper for Autogluon TimeSeriesPredictor.

    This class handles the conversion of input data formats (pandas DataFrame or Avro bytes)
    into the TimeSeriesDataFrame format required by Autogluon for inference.
    """
    def __init__(self, predictor: TimeSeriesPredictor):
        self.predictor = predictor

    def predict(self, context, model_input: pd.DataFrame):
        """
        Predicts future values using the wrapped Autogluon TimeSeriesPredictor.

        Args:
            context: MLflow context (not used, but required by PyFunc).
            model_input: The input data, which can be a pandas DataFrame or raw Avro bytes.

        Returns:
            The prediction result from the Autogluon predictor.
        """
        if isinstance(model_input, (bytes, io.BytesIO)):
            if isinstance(model_input, bytes):
                input_stream = io.BytesIO(model_input)
            else:
                input_stream = model_input
            
            # Read Avro bytes from the stream and convert to a pandas DataFrame
            reader = fastavro.reader(input_stream)
            records = [r for r in reader]
            df = pd.DataFrame.from_records(records)
        elif isinstance(model_input, pd.DataFrame):
            # If input is already a DataFrame, use it directly
            df = model_input
        else:
            raise TypeError("Input must be a pandas DataFrame or raw Avro bytes.")

        # Convert the pandas DataFrame to an Autogluon TimeSeriesDataFrame
        ts_dataframe = TimeSeriesDataFrame(df)
        return self.predictor.predict(ts_dataframe)

def log_autogluon_timeseries_to_mlflow_artifact(predictor: TimeSeriesPredictor, input_example: pd.DataFrame):
    """
    Logs the Autogluon TimeSeriesPredictor to MLflow using a custom PythonModel wrapper.

    This function wraps the predictor in `AGTimeSeriesWrapper` to ensure the MLflow deployment
    can handle different input formats (like Avro). It also specifies the Conda environment
    required for a GPU-enabled deployment.

    Args:
        predictor: The trained Autogluon TimeSeriesPredictor instance.
        input_example: A sample DataFrame to be saved with the model for schema enforcement.
    """
    # Define the Conda environment required for GPU inference
    conda_env = {
            "channels": ["defaults", "conda-forge", "pytorch"],
            "dependencies": [
                "python=3.10", # Specify the exact Python version
                "pip",
                "pytorch=2.2.0=py3.10_cuda11.8_cudnn8.7_0", # Match the image version precisely
                "pytorch-cuda=11.8",
                "autogluon.timeseries",
                "fastavro",
                {"pip": [
                    "mlflow",
                    "cloudpickle",
                    "sagemaker-mlflow",
                ]}
            ]
        }

    # Log the custom PyFunc model artifact and return the logged model object
    logged_model = log_model(
        python_model=AGTimeSeriesWrapper(predictor),
        artifact_path="model",  # Use artifact_path instead of name
        conda_env=conda_env,
        input_example=input_example
    )

    print(f"Model logged as artifact under name 'model'")
    return logged_model


def _safe(v, cast=float):
    """
    Safely casts a value to a specified type, returning None if an error occurs.
    """
    try:
        return cast(v)
    except Exception:
        return None

# ----------------------------
# File discovery (recursive)
# ----------------------------
PARQUET_EXTS = {".parquet", ".PARQUET", ".pq", ".PQ"}

def _find_parquet_files(root: str, keyword: str | None):
    """
    Recursively finds all Parquet files in a directory, optionally filtering by a keyword.
    """
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"Data directory not found: {root}")
    files = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            _, ext = os.path.splitext(fn)
            if ext in PARQUET_EXTS:
                if keyword:
                    if keyword in fn:
                        files.append(os.path.join(r, fn))
                else:
                    files.append(os.path.join(r, fn))
    print(f"[finder] root={root} keyword={keyword!r} found={len(files)}")
    if not files:
        raise FileNotFoundError(f"No parquet files in {root} (keyword={keyword!r})")
    return sorted(files)

# ----------------------------
# Loader -> TSF objects (target + optional covariates)
# ----------------------------
@retry_decorator(max_attempts=3, delay_seconds=30, backoff_factor=2)
def load_timeseries_parquet(
    data_root: str,
    keyword: str | None,
    id_col: str,
    time_col: str,
    target_col: str,
    covariate_cols: list[str] | None = None,
):
    """
    Loads Parquet files, combines them, and converts the data into TimeSeriesDataFrame objects
    for both the target variable and optional covariates.
    """
    files = _find_parquet_files(data_root, keyword)

    # Helper function to find the correct column name from a set of aliases (currently unused)
    def resolve(cols: list[str], desired: str, aliases: list[str]) -> str | None:
        norm = {c.strip().lower(): c for c in cols}
        for cand in [desired] + aliases:
            k = cand.strip().lower()
            if k in norm:
                return norm[k]
        return None

    frames = []
    for fp in files:
        t = pq.read_table(fp)
        df = t.to_pandas()
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True).sort_values([id_col, time_col]).reset_index(drop=True)

    # Create a TimeSeriesDataFrame for the target variable
    target_tsf = TimeSeriesDataFrame.from_data_frame(
        all_df[[id_col, time_col, target_col]],
        id_column=id_col,
        timestamp_column=time_col,
    )

    cov_tsf = None
    if covariate_cols:
        # Create a TimeSeriesDataFrame for covariates if they exist
        present = [c for c in covariate_cols if c in all_df.columns]
        if present:
            cov_df = all_df[[id_col, time_col] + present]
            cov_tsf = TimeSeriesDataFrame.from_data_frame(cov_df, id_column=id_col, timestamp_column=time_col)

    return target_tsf, cov_tsf

def log_autogluon_timeseries_metrics(predictor: TimeSeriesPredictor, run_params: dict | None = None):
    """
    Logs metrics, artifacts, and parameters from the Autogluon TimeSeriesPredictor to MLflow.

    Args:
        predictor: The trained Autogluon TimeSeriesPredictor instance.
        run_params: Optional dictionary of additional parameters to log.
    """
    lb = predictor.leaderboard(silent=True)
    
    # Log the leaderboard as CSV and JSON artifacts
    with tempfile.TemporaryDirectory() as td:
        lb_csv = os.path.join(td, "leaderboard.csv")
        lb_json = os.path.join(td, "leaderboard.json")
        lb.to_csv(lb_csv, index=False)
        lb.to_json(lb_json, orient="records")
        mlflow.log_artifact(lb_csv, artifact_path="leaderboard")
        mlflow.log_artifact(lb_json, artifact_path="leaderboard")
        try:
            if hasattr(mlflow, "log_table"):
                mlflow.log_table(lb, artifact_file="leaderboard/leaderboard_table.json")
        except Exception:
            pass

    # Log the best model's name and validation score
    score_col = next((c for c in ["score_val", "score_val_wo_nan", "score"] if c in lb.columns), None)
    if score_col is not None and not lb.empty:
        # Use iloc[0] to get the best-scoring row
        best_row = lb.sort_values(score_col, ascending=False).iloc[0] # Note: Autogluon default is lower is better, so asc=True is typical
        mlflow.log_param("best_model", best_row["model"])
        mlflow.log_metric("best_model_val_score", _safe(best_row[score_col]))
    else:
        mlflow.log_param("best_model", predictor.get_model_best())

    # Log metrics and parameters for each individual model
    for m in predictor.model_names():
        try:
            mdl = predictor._trainer.load_model(m)
            fit_time = getattr(mdl, "fit_time", None)
            val_score = getattr(mdl, "val_score", None)
            if fit_time is not None:
                mlflow.log_metric(f"{m}_fit_time", _safe(fit_time))
            if val_score is not None:
                mlflow.log_metric(f"{m}_val_score", _safe(val_score))

            params = getattr(mdl, "params", {}) or {}
            epochs = None
            for k in ["num_epochs", "epochs", "max_epochs", "max_num_epochs"]:
                if k in params:
                    epochs = _safe(params[k], int)
                    break
            if epochs is not None:
                mlflow.log_metric(f"{m}_epochs", epochs)

            with tempfile.TemporaryDirectory() as td:
                pth = os.path.join(td, f"{m}_params.json")
                with open(pth, "w") as f:
                    json.dump(params, f, indent=2, default=str)
                mlflow.log_artifact(pth, artifact_path=f"models/{m}")

            hist = getattr(mdl, "training_history", None)
            if hist is not None:
                try:
                    hist_df = pd.DataFrame(hist)
                except Exception:
                    hist_df = pd.DataFrame(hist) if isinstance(hist, dict) else None
                if hist_df is not None and not hist_df.empty:
                    with tempfile.TemporaryDirectory() as td:
                        hp = os.path.join(td, f"{m}_training_history.csv")
                        hist_df.to_csv(hp, index=False)
        except Exception as e:
            print(f"Failed to log metrics for model {m}: {e}")



## Inference

def package_mlflow_model(run_id, bucket, prefix, mlflow_tracking_arn, artifact_path="model"):
    """
    Packages an MLflow model from an S3 artifact directory into a tar.gz file
    and uploads it to a specified S3 bucket and prefix.

    Args:
        run_id (str): The ID of the MLflow run.
        bucket (str): The name of the S3 bucket to upload to.
        prefix (str): The S3 key prefix (folder) within the bucket.
        artifact_path (str): The path to the model artifact within the run.

    Returns:
        str: The S3 URI of the newly created tar.gz model archive.
    """
    mlflow.set_tracking_uri(mlflow_tracking_arn)

    client = MlflowClient()

    logged_model_uri = f"runs:/{run_id}/{artifact_path}"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download the model artifacts to a temporary directory
        mlflow.artifacts.download_artifacts(
            artifact_uri=logged_model_uri,
            dst_path=tmp_dir
        )

        # Create a tar.gz archive from the downloaded directory
        tar_gz_path = os.path.join(tmp_dir, "model.tar.gz")
        with tarfile.open(tar_gz_path, "w:gz") as tar:
            tar.add(os.path.join(tmp_dir, artifact_path), arcname=artifact_path)
            
        # Upload the tar.gz to S3 using the SageMaker Session
        sagemaker_session = sagemaker.Session()
        s3_uri = sagemaker_session.upload_data(
            path=tar_gz_path,
            bucket=bucket,        # Explicitly specify the bucket
            key_prefix=prefix,    # Explicitly specify the key prefix
        )
        
        return s3_uri
