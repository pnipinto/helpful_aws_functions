import json, pandas as pd, mlflow, mlflow.pyfunc
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

class AGTimeSeriesWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = TimeSeriesPredictor.load(context.artifacts["predictor"])
        try:
            with open(f'{context.artifacts["predictor"]}/exog_config.json', "r") as f:
                self.cfg = json.load(f)
        except Exception:
            self.cfg = {"feature_name": None, "feature_role": "none", "feature_fill": "ffill"}

    def _extend_known(self, cov_tsf, horizon, fill):
        pdf = cov_tsf.to_pandas().sort_values(["item_id","timestamp"])
        v = pdf.columns.difference(["item_id","timestamp"])[0]
        out = []
        for item, g in pdf.groupby("item_id"):
            if len(g) < 2: raise ValueError(f"Need ≥2 points to infer freq for {item}")
            freq = g["timestamp"].iloc[1] - g["timestamp"].iloc[0]
            fut_idx = pd.date_range(g["timestamp"].max() + freq, periods=horizon, freq=freq)
            if fill == "ffill": vals = [g[v].iloc[-1]] * horizon
            elif fill == "zero": vals = [0.0] * horizon
            elif fill == "mean": vals = [float(g[v].mean())] * horizon
            else: vals = [g[v].iloc[-1]] * horizon
            fut = pd.DataFrame({"item_id": item, "timestamp": fut_idx, v: vals})
            out.append(pd.concat([g, fut], ignore_index=True))
        from autogluon.timeseries import TimeSeriesDataFrame
        return TimeSeriesDataFrame.from_data_frame(pd.concat(out, ignore_index=True),
                                                   id_column="item_id", timestamp_column="timestamp")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        df = model_input.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cols = [c for c in ["item_id","timestamp","target"] if c in df.columns]
        tsf = TimeSeriesDataFrame.from_data_frame(df[cols], id_column="item_id", timestamp_column="timestamp")
        feature = self.cfg.get("feature_name")
        role    = self.cfg.get("feature_role", "none")
        fill    = self.cfg.get("feature_fill", "ffill")
        known = past = static = None
        if feature and feature in df.columns:
            cov = TimeSeriesDataFrame.from_data_frame(df[["item_id","timestamp",feature]],
                                                      id_column="item_id", timestamp_column="timestamp")
            if role == "past": past = cov
            elif role == "known": known = self._extend_known(cov, self.predictor.prediction_length, fill)
            elif role == "static":
                static = df.groupby("item_id")[feature].last().to_frame(name=feature); static.index.name = "item_id"
        fcst = self.predictor.predict(tsf, known_covariates=known, past_covariates=past, static_features=static)
        return fcst.to_pandas().reset_index()
