import os, sys, time, glob, argparse
from functools import wraps
import json # Used in helper functions, but good practice to keep here
import pandas as pd
import pyarrow.parquet as pq
import mlflow
import cloudpickle

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Import your helper functions from the local file
from helper_functions import (
    AGTimeSeriesWrapper,
    load_timeseries_parquet,
    log_autogluon_timeseries_to_mlflow_artifact,
    log_autogluon_timeseries_metrics,
)

# ----------------------------
# Main
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


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment)

    mlflow.autolog(disable=True)

    print(f"[load] train_dir={args.train_dir} keyword={args.train_keyword!r}")
    train_tsf, train_cov_tsf = load_timeseries_parquet(
        args.train_dir, args.train_keyword, args.id_col, args.time_col, args.target_col,
        covariate_cols=["random_feature"],
    )

    try:
        test_tsf, test_cov_tsf = load_timeseries_parquet(
            args.test_dir, args.test_keyword, args.id_col, args.time_col, args.target_col,
            covariate_cols=["random_feature"],
        )
    except FileNotFoundError:
        test_tsf, test_cov_tsf = None, None

    with mlflow.start_run() as run:
        # Log training parameters
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

        # Train the AutoGluon model
        predictor = TimeSeriesPredictor(
            prediction_length=args.prediction_length,
            eval_metric=args.eval_metric,
            path=args.output_dir,
            # num_gpus = args.num_gpus,
        )
        predictor.fit(
            train_data=train_tsf,
            presets=args.presets,
            time_limit=args.time_limit,
        )
        predictor.save()

        # Step 1: Log the model artifact and capture the returned object
        # Use the original DataFrame structure for the input example
        input_example_df = train_tsf.head(10).to_pandas()
        print("--- Debugging input_example_df ---")
        print("Input Example DataFrame head:")
        print(input_example_df.head())
        print("Input Example DataFrame columns:")
        print(input_example_df.columns)
        print("---------------------------------")
        
        logged_model = log_autogluon_timeseries_to_mlflow_artifact(predictor, input_example_df)
        
        # Log additional AutoGluon metrics and details
        log_autogluon_timeseries_metrics(predictor)

        if test_tsf is not None:
            scores = predictor.evaluate(test_tsf)
            for k, v in scores.items():
                mlflow.log_metric(f"test_{k}", float(v))
        
        # Step 2: Explicitly register the model using the URI from the logged object
        mlflow.register_model(model_uri=logged_model.model_uri, name=args.model_name)

        print("[done] training complete and model logged and registered to MLflow.")

if __name__ == "__main__":
    main()
