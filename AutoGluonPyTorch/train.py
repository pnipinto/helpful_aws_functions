import os, sys, time, glob, argparse
from functools import wraps

import pandas as pd
import pyarrow.parquet as pq
import mlflow

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from helper_functions import AGTimeSeriesWrapper  # keep this file in source_dir

# ----------------------------
# Retry helper
# ----------------------------
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

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=str, default='/opt/ml/model')

    # MLflow (managed tracking server ARN, like your tabular script)
    p.add_argument('--mlflow_arn', type=str, required=True)
    p.add_argument('--mlflow_experiment', type=str, required=True)

    # Data roots (default to SageMaker channels)
    p.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    p.add_argument('--test-dir',  type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))  # empty means "no test"

    # Optional filename filters (empty -> load all)
    p.add_argument('--train-keyword', type=str, default=None)
    p.add_argument('--test-keyword',  type=str, default=None)

    # Schema
    p.add_argument('--id-col', type=str, default='item_id')
    p.add_argument('--time-col', type=str, default='timestamp')
    p.add_argument('--target-col', type=str, default='target')

    # Model
    p.add_argument('--prediction-length', type=int, default=24)
    p.add_argument('--eval-metric', type=str, default='MAPE')
    p.add_argument('--presets', type=str, default='best_quality')
    p.add_argument('--time-limit', type=int, default=900)
    p.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', '0')))
    return p.parse_args()

# ----------------------------
# File discovery (recursive)
# ----------------------------
PARQUET_EXTS = {".parquet", ".PARQUET", ".pq", ".PQ"}

def _find_parquet_files(root: str, keyword: str | None):
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
    covariate_cols: list[str] | None = None,  # e.g., ["random_feature"]
):
    files = _find_parquet_files(data_root, keyword)

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

    all_df = pd.concat(frames, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    target_tsf = TimeSeriesDataFrame.from_data_frame(
        all_df[["item_id", "timestamp", "target"]],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    cov_tsf = None
    if covariate_cols:
        present = [c for c in covariate_cols if c in all_df.columns]  # (only if you merged covs into all_df)
        if present:
            cov_df = all_df[["item_id", "timestamp"] + present]
            cov_tsf = TimeSeriesDataFrame.from_data_frame(cov_df, id_column="item_id", timestamp_column="timestamp")

    return target_tsf, cov_tsf

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # MLflow (managed server). Requires sagemaker-mlflow installed in the container.
    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment)

    # TRAIN
    print(f"[load] train_dir={args.train_dir} keyword={args.train_keyword!r}")
    train_tsf, train_cov_tsf = load_timeseries_parquet(
        args.train_dir, args.train_keyword, args.id_col, args.time_col, args.target_col,
        covariate_cols=["random_feature"],
    )

    # TEST (optional)
    test_tsf = test_cov_tsf = None
    if args.test_dir and os.path.isdir(args.test_dir):
        print(f"[load] test_dir={args.test_dir} keyword={args.test_keyword!r}")
        try:
            test_tsf, test_cov_tsf = load_timeseries_parquet(
                args.test_dir, args.test_keyword, args.id_col, args.time_col, args.target_col,
                covariate_cols=["random_feature"],
            )
        except Exception as e:
            print(f"[load] test skipped: {e}")

    with mlflow.start_run():
        mlflow.log_params({
            "prediction_length": args.prediction_length,
            "eval_metric": args.eval_metric,
            "presets": args.presets,
            "time_limit": args.time_limit,
            "train_dir": args.train_dir,
            "test_dir": args.test_dir,
            "train_keyword": args.train_keyword,
            "test_keyword": args.test_keyword,
        })

        predictor = TimeSeriesPredictor(
            prediction_length=args.prediction_length,
            eval_metric=args.eval_metric,
            path=args.output_dir,
        )

        predictor.fit(
            train_data=train_tsf,
            past_covariates=train_cov_tsf,    # random_feature as past covariate (change if you want 'known' or 'static')
            presets=args.presets,
            time_limit=args.time_limit,
            num_gpus=args.num_gpus,
        )
        predictor.save()

        if test_tsf is not None:
            scores = predictor.evaluate(test_tsf, past_covariates=test_cov_tsf)
            for k, v in scores.items():
                mlflow.log_metric(f"test_{k}", float(v))

        # Deployable PyFunc
        conda_env = {
            "name": "agts-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                {"pip": [
                    "autogluon.timeseries[all]==1.1.1",
                    "pandas>=2.0.0",
                    "pyarrow>=13.0.0",
                    "mlflow>=2.9.0",
                    "sagemaker-mlflow>=0.1.0",
                ]},
            ],
        }
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=AGTimeSeriesWrapper(),
            artifacts={"predictor": args.output_dir},
            conda_env=conda_env,
        )
        print("[done] training complete and model logged to MLflow.")

if __name__ == "__main__":
    main()
