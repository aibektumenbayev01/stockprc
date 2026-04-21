#!/usr/bin/env python3
"""
Generate enhanced model-comparison graphics for NEE with 80/10/10 split.

This version adds:
  - strict train/validation/test protocol (80/10/10),
  - external market features (SPY, VIX),
  - company-size and ownership-related features
    (shares outstanding, market cap, institutional/insider ownership proxies),
  - validation-based tuning for ML models.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pypdf import PdfReader
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

try:
    import torch
    from chronos import ChronosPipeline

    HAS_CHRONOS = True
except Exception:
    HAS_CHRONOS = False


ROW_PATTERN = re.compile(
    r"^\s*"
    r"(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<price>\d+(?:\.\d+)?)\s+"
    r"(?P<open>\d+(?:\.\d+)?)\s+"
    r"(?P<high>\d+(?:\.\d+)?)\s+"
    r"(?P<low>\d+(?:\.\d+)?)\s+"
    r"(?P<volume>\d+(?:\.\d+)?[KMB])\s+"
    r"(?P<change>[+-]?\d+(?:\.\d+)?)%"
    r"\s*$"
)

COMPACT_ROW_PATTERN = re.compile(
    r"^\s*"
    r"(?P<date>\d{2}/\d{2}/\d{4})"
    r"(?P<price>\d{2,3}\.\d{2})"
    r"(?P<open>\d{2,3}\.\d{2})"
    r"(?P<high>\d{2,3}\.\d{2})"
    r"(?P<low>\d{2,3}\.\d{2})"
    r"(?P<volume>\d{1,3}\.\d{2}[KMB])"
    r"(?P<change>[+-]?\d{1,2}\.\d{2})%"
    r"\s*$"
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def parse_volume(volume_text: str) -> float:
    suffix = volume_text[-1]
    value = float(volume_text[:-1])
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers.get(suffix, 1.0)


def _normalize_date_index(index_like: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index_like)
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_localize(None)
    return pd.DatetimeIndex(dt_index).normalize()


def extract_prices_from_pdf(pdf_path: Path) -> pd.DataFrame:
    reader = PdfReader(str(pdf_path))
    rows: List[Dict[str, float]] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            match = ROW_PATTERN.match(line)
            if not match:
                match = COMPACT_ROW_PATTERN.match(line)
            if not match:
                continue
            rows.append(
                {
                    "Date": pd.to_datetime(match.group("date"), format="%m/%d/%Y"),
                    "Price": float(match.group("price")),
                    "Open": float(match.group("open")),
                    "High": float(match.group("high")),
                    "Low": float(match.group("low")),
                    "Volume": parse_volume(match.group("volume")),
                    "ChangePct": float(match.group("change")),
                }
            )

    if not rows:
        raise RuntimeError(f"No data rows extracted from PDF: {pdf_path}")

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return df


def download_market_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(symbol).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        interval="1d",
    )
    if hist.empty:
        return pd.Series(dtype=float)
    series = hist["Close"].copy()
    series.index = _normalize_date_index(series.index)
    return series.groupby(level=0).last().sort_index()


def build_ownership_proxy(
    ticker: yf.Ticker, dates: pd.Series, shares_outstanding: pd.Series
) -> pd.DataFrame:
    out = pd.DataFrame({"Date": dates})
    out["insider_net_shares"] = 0.0

    try:
        tx = ticker.insider_transactions
    except Exception:
        tx = None

    if isinstance(tx, pd.DataFrame) and not tx.empty:
        tx = tx.copy()
        date_col = next((c for c in tx.columns if "date" in c.lower()), None)
        shares_col = next((c for c in tx.columns if "share" in c.lower()), None)
        desc_col = next(
            (c for c in tx.columns if c.lower() in ("text", "transaction")), None
        )

        if date_col and shares_col:
            tx = tx[[date_col, shares_col] + ([desc_col] if desc_col else [])].copy()
            tx[date_col] = pd.to_datetime(tx[date_col], errors="coerce")
            tx = tx.dropna(subset=[date_col])
            tx["Date"] = _normalize_date_index(tx[date_col])
            tx[shares_col] = (
                tx[shares_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
                .abs()
            )

            if desc_col:
                text = tx[desc_col].astype(str).str.lower()
                sign = np.where(
                    text.str.contains("sale|sell"),
                    -1.0,
                    np.where(text.str.contains("buy|purchase|acquire"), 1.0, 0.0),
                )
                tx["signed_shares"] = tx[shares_col] * sign
            else:
                tx["signed_shares"] = 0.0

            daily_net = tx.groupby("Date")["signed_shares"].sum()
            out = out.merge(
                daily_net.rename("insider_net_shares"), how="left", on="Date"
            )
            out["insider_net_shares"] = out["insider_net_shares_y"].fillna(
                out["insider_net_shares_x"]
            )
            out = out.drop(columns=["insider_net_shares_x", "insider_net_shares_y"])

    out["insider_net_shares"] = out["insider_net_shares"].fillna(0.0)
    out["insider_ownership_delta_pct"] = (
        out["insider_net_shares"] / shares_outstanding.replace(0, np.nan).values
    ).fillna(0.0)
    out["insider_ownership_cum_proxy_pct"] = out["insider_ownership_delta_pct"].cumsum()
    out["insider_ownership_20d_change_pct"] = (
        out["insider_ownership_cum_proxy_pct"]
        - out["insider_ownership_cum_proxy_pct"].shift(20).fillna(0.0)
    )
    return out


def enrich_with_external_and_fundamental_features(df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    out = df.copy()
    start = out["Date"].min() - pd.Timedelta(days=60)
    end = out["Date"].max() + pd.Timedelta(days=3)

    spy_close = download_market_series("SPY", start, end)
    vix_close = download_market_series("^VIX", start, end)

    out = out.merge(
        pd.DataFrame({"Date": spy_close.index, "SPY_Close": spy_close.values}),
        on="Date",
        how="left",
    )
    out = out.merge(
        pd.DataFrame({"Date": vix_close.index, "VIX_Close": vix_close.values}),
        on="Date",
        how="left",
    )

    ticker = yf.Ticker(ticker_symbol)
    info = ticker.get_info()
    shares_const = float(info.get("sharesOutstanding") or np.nan)
    market_cap_const = float(info.get("marketCap") or np.nan)
    insider_pct_const = float(info.get("heldPercentInsiders") or 0.0)
    inst_pct_const = float(info.get("heldPercentInstitutions") or 0.0)

    shares_series = None
    try:
        shares_full = ticker.get_shares_full(start=start.strftime("%Y-%m-%d"))
    except Exception:
        shares_full = None

    if isinstance(shares_full, pd.Series) and not shares_full.empty:
        shares_series = shares_full.copy()
        shares_series.index = _normalize_date_index(shares_series.index)
        shares_series = shares_series.groupby(level=0).last().sort_index()

    if shares_series is None:
        out["shares_outstanding"] = shares_const
    else:
        shares_df = pd.DataFrame(
            {"Date": shares_series.index, "shares_outstanding": shares_series.values}
        )
        out = out.merge(shares_df, how="left", on="Date")
        out["shares_outstanding"] = out["shares_outstanding"].ffill().bfill()
        if np.isnan(out["shares_outstanding"]).any():
            out["shares_outstanding"] = out["shares_outstanding"].fillna(shares_const)

    out["shares_outstanding"] = out["shares_outstanding"].astype(float)
    out["market_cap"] = out["Price"] * out["shares_outstanding"]

    if np.isnan(out["market_cap"]).all():
        out["market_cap"] = market_cap_const

    out["inst_ownership_pct"] = inst_pct_const
    out["insider_ownership_pct"] = insider_pct_const

    try:
        inst_holders = ticker.institutional_holders
    except Exception:
        inst_holders = None

    if isinstance(inst_holders, pd.DataFrame) and not inst_holders.empty:
        cols = {c.lower(): c for c in inst_holders.columns}
        date_col = cols.get("date reported")
        pct_col = next((c for c in inst_holders.columns if "pct" in c.lower()), None)
        if date_col and pct_col:
            temp = inst_holders[[date_col, pct_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col])
            temp["Date"] = _normalize_date_index(temp[date_col])
            temp[pct_col] = pd.to_numeric(temp[pct_col], errors="coerce")
            inst_by_date = temp.groupby("Date")[pct_col].sum().clip(lower=0.0, upper=1.5)
            out = out.merge(
                pd.DataFrame(
                    {
                        "Date": inst_by_date.index,
                        "inst_ownership_pct_reported": inst_by_date.values,
                    }
                ),
                how="left",
                on="Date",
            )
            out["inst_ownership_pct_reported"] = (
                out["inst_ownership_pct_reported"].ffill().bfill()
            )
            out["inst_ownership_pct"] = out["inst_ownership_pct_reported"].fillna(
                out["inst_ownership_pct"]
            )
            out = out.drop(columns=["inst_ownership_pct_reported"])

    ownership_proxy = build_ownership_proxy(
        ticker=ticker, dates=out["Date"], shares_outstanding=out["shares_outstanding"]
    )
    out = out.merge(ownership_proxy, how="left", on="Date")

    out["SPY_Close"] = out["SPY_Close"].ffill().bfill()
    out["VIX_Close"] = out["VIX_Close"].ffill().bfill()

    out["market_cap"] = out["market_cap"].ffill().bfill()
    out["inst_ownership_pct"] = out["inst_ownership_pct"].ffill().bfill()
    out["insider_ownership_pct"] = out["insider_ownership_pct"].ffill().bfill()
    out["turnover_ratio"] = (
        out["Volume"] / out["shares_outstanding"].replace(0, np.nan)
    ).fillna(0.0)

    out["SPY_ret_1"] = out["SPY_Close"].pct_change(1)
    out["SPY_ret_5"] = out["SPY_Close"].pct_change(5)
    out["VIX_ret_1"] = out["VIX_Close"].pct_change(1)
    out["VIX_ret_5"] = out["VIX_Close"].pct_change(5)
    out["price_ret_1"] = out["Price"].pct_change(1)
    out["price_realized_vol_10"] = out["price_ret_1"].rolling(10).std()
    out["intraday_spread"] = (out["High"] - out["Low"]) / out["Price"].replace(0, np.nan)
    out["ownership_gap_pct"] = out["inst_ownership_pct"] - out["insider_ownership_pct"]

    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    lag_windows = [1, 2, 3, 5, 10, 20]
    roll_windows = [5, 10, 20]

    for lag in lag_windows:
        feat[f"price_lag_{lag}"] = feat["Price"].shift(lag)

    for window in roll_windows:
        feat[f"price_roll_mean_{window}"] = feat["Price"].shift(1).rolling(window).mean()
        feat[f"price_roll_std_{window}"] = feat["Price"].shift(1).rolling(window).std()
        feat[f"volume_roll_mean_{window}"] = feat["Volume"].shift(1).rolling(window).mean()
        feat[f"vix_roll_mean_{window}"] = feat["VIX_Close"].shift(1).rolling(window).mean()

    base_predictors = [
        "Open",
        "High",
        "Low",
        "Volume",
        "ChangePct",
        "SPY_Close",
        "VIX_Close",
        "SPY_ret_1",
        "SPY_ret_5",
        "VIX_ret_1",
        "VIX_ret_5",
        "market_cap",
        "shares_outstanding",
        "inst_ownership_pct",
        "insider_ownership_pct",
        "insider_ownership_20d_change_pct",
        "turnover_ratio",
        "price_ret_1",
        "price_realized_vol_10",
        "intraday_spread",
        "ownership_gap_pct",
    ]

    for col in base_predictors:
        feat[f"{col}_lag1"] = feat[col].shift(1)

    return feat.dropna().reset_index(drop=True)


def evaluate_prediction_dict(
    actual_df: pd.DataFrame, prediction_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    actual_series = actual_df.set_index("Date")["Price"]
    rows = []

    for model_name, pred_df in prediction_dict.items():
        pred_s = pred_df.set_index("Date")["Pred"]
        common_idx = actual_series.index.intersection(pred_s.index)
        y_true = actual_series.loc[common_idx].to_numpy()
        y_pred = pred_s.loc[common_idx].to_numpy()
        rows.append(
            {
                "Model": model_name,
                "Rows": int(len(common_idx)),
                "MAE": round(mae(y_true, y_pred), 4),
                "RMSE": round(rmse(y_true, y_pred), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def rolling_chronos_predict(
    pipeline: ChronosPipeline,
    full_prices: np.ndarray,
    full_dates: np.ndarray,
    start_idx: int,
    end_idx: int,
    base_seed: int,
) -> pd.DataFrame:
    preds: List[float] = []
    dates = full_dates[start_idx:end_idx]
    for i in range(start_idx, end_idx):
        torch.manual_seed(base_seed + i)
        context_start = max(0, i - 512)
        context = torch.tensor(full_prices[context_start:i], dtype=torch.float32)
        samples = pipeline.predict(context, prediction_length=1, num_samples=40)
        pred = samples.median(dim=1).values[0, 0].detach().cpu().item()
        preds.append(float(pred))
    return pd.DataFrame({"Date": dates, "Pred": preds})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced NEE model comparison.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/workspace/data/nee_prices_clean.csv"),
        help="Fallback CSV with Date/Price/Open/High/Low/Volume/ChangePct if PDF is unavailable.",
    )
    parser.add_argument(
        "--input-pdf",
        type=Path,
        default=Path(
            "/home/ubuntu/.cursor/projects/workspace/uploads/NextEra_Energy_Stock_Price_History-4.pdf"
        ),
    )
    parser.add_argument("--ticker", type=str, default="NEE")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/results"))
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chronos-model", type=str, default="amazon/chronos-t5-tiny")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and enrich data.
    if args.input_pdf.exists():
        raw_df = extract_prices_from_pdf(args.input_pdf)
    elif args.input_csv.exists():
        raw_df = pd.read_csv(args.input_csv, parse_dates=["Date"]).copy()
        required = {"Date", "Price", "Open", "High", "Low", "Volume", "ChangePct"}
        missing = required.difference(raw_df.columns)
        if missing:
            raise RuntimeError(
                f"Input CSV is missing required columns: {sorted(missing)}"
            )
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)
    else:
        raise RuntimeError(
            f"Neither input PDF nor fallback CSV exists. "
            f"Checked: {args.input_pdf} and {args.input_csv}"
        )
    enriched_df = enrich_with_external_and_fundamental_features(raw_df, args.ticker)
    enriched_df = enriched_df.sort_values("Date").reset_index(drop=True)

    cleaned_csv = args.data_dir / "nee_prices_clean.csv"
    enriched_csv = args.data_dir / "nee_prices_enriched.csv"
    raw_df.to_csv(cleaned_csv, index=False)
    enriched_df.to_csv(enriched_csv, index=False)

    # 2) Strict chronological 80/10/10 split.
    n = len(enriched_df)
    train_end_idx = int(n * 0.8)
    val_end_idx = int(n * 0.9)

    train_df = enriched_df.iloc[:train_end_idx].copy()
    val_df = enriched_df.iloc[train_end_idx:val_end_idx].copy()
    test_df = enriched_df.iloc[val_end_idx:].copy()
    full_prices = enriched_df["Price"].to_numpy()
    full_dates = enriched_df["Date"].to_numpy()

    val_predictions: Dict[str, pd.DataFrame] = {}
    test_predictions: Dict[str, pd.DataFrame] = {}

    # 3) Time-series baselines: validation first, then retrain on train+val for test.
    # Naive
    naive_val = enriched_df["Price"].shift(1).iloc[train_end_idx:val_end_idx].to_numpy()
    naive_test = enriched_df["Price"].shift(1).iloc[val_end_idx:].to_numpy()
    val_predictions["Naive"] = pd.DataFrame({"Date": val_df["Date"], "Pred": naive_val})
    test_predictions["Naive"] = pd.DataFrame({"Date": test_df["Date"], "Pred": naive_test})

    # ARIMA
    arima_val_fit = ARIMA(train_df["Price"].to_numpy(), order=(5, 1, 0)).fit()
    arima_val_pred = np.asarray(arima_val_fit.forecast(steps=len(val_df)))
    val_predictions["ARIMA"] = pd.DataFrame(
        {"Date": val_df["Date"], "Pred": arima_val_pred}
    )
    arima_test_fit = ARIMA(
        enriched_df.iloc[:val_end_idx]["Price"].to_numpy(), order=(5, 1, 0)
    ).fit()
    arima_test_pred = np.asarray(arima_test_fit.forecast(steps=len(test_df)))
    test_predictions["ARIMA"] = pd.DataFrame(
        {"Date": test_df["Date"], "Pred": arima_test_pred}
    )

    # ETS
    ets_val_fit = ExponentialSmoothing(
        train_df["Price"].to_numpy(), trend="add", damped_trend=True
    ).fit(optimized=True)
    ets_val_pred = np.asarray(ets_val_fit.forecast(len(val_df)))
    val_predictions["ETS"] = pd.DataFrame({"Date": val_df["Date"], "Pred": ets_val_pred})
    ets_test_fit = ExponentialSmoothing(
        enriched_df.iloc[:val_end_idx]["Price"].to_numpy(), trend="add", damped_trend=True
    ).fit(optimized=True)
    ets_test_pred = np.asarray(ets_test_fit.forecast(len(test_df)))
    test_predictions["ETS"] = pd.DataFrame({"Date": test_df["Date"], "Pred": ets_test_pred})

    # 4) Feature-based models with validation tuning.
    feat_df = make_features(enriched_df)
    train_end_date = train_df["Date"].max()
    val_end_date = val_df["Date"].max()

    train_feat = feat_df[feat_df["Date"] <= train_end_date].copy()
    val_feat = feat_df[
        (feat_df["Date"] > train_end_date) & (feat_df["Date"] <= val_end_date)
    ].copy()
    test_feat = feat_df[feat_df["Date"] > val_end_date].copy()

    feature_cols = [c for c in feat_df.columns if c not in ("Date", "Price")]

    X_train = train_feat[feature_cols].to_numpy()
    y_train = train_feat["Price"].to_numpy()
    X_val = val_feat[feature_cols].to_numpy()
    y_val = val_feat["Price"].to_numpy()
    X_train_val = pd.concat([train_feat, val_feat], axis=0)[feature_cols].to_numpy()
    y_train_val = pd.concat([train_feat, val_feat], axis=0)["Price"].to_numpy()
    X_test = test_feat[feature_cols].to_numpy()

    # Random Forest tuning
    rf_grid = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 1},
        {"n_estimators": 700, "max_depth": 8, "min_samples_leaf": 2},
    ]
    best_rf_cfg = None
    best_rf_rmse = float("inf")
    for cfg in rf_grid:
        rf_model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=args.random_seed,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        pred = rf_model.predict(X_val)
        score = rmse(y_val, pred)
        if score < best_rf_rmse:
            best_rf_rmse = score
            best_rf_cfg = cfg

    rf_val = RandomForestRegressor(
        **best_rf_cfg, random_state=args.random_seed, n_jobs=-1
    ).fit(X_train, y_train)
    val_predictions["RandomForest"] = pd.DataFrame(
        {"Date": val_feat["Date"].to_numpy(), "Pred": rf_val.predict(X_val)}
    )

    rf_test = RandomForestRegressor(
        **best_rf_cfg, random_state=args.random_seed, n_jobs=-1
    ).fit(X_train_val, y_train_val)
    test_predictions["RandomForest"] = pd.DataFrame(
        {"Date": test_feat["Date"].to_numpy(), "Pred": rf_test.predict(X_test)}
    )

    # XGBoost tuning
    xgb_grid = [
        {"n_estimators": 350, "learning_rate": 0.03, "max_depth": 3},
        {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4},
        {"n_estimators": 450, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 700, "learning_rate": 0.02, "max_depth": 4},
    ]
    common_xgb = {
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": args.random_seed,
    }
    best_xgb_cfg = None
    best_xgb_rmse = float("inf")
    for cfg in xgb_grid:
        xgb_model = XGBRegressor(**cfg, **common_xgb)
        xgb_model.fit(X_train, y_train)
        pred = xgb_model.predict(X_val)
        score = rmse(y_val, pred)
        if score < best_xgb_rmse:
            best_xgb_rmse = score
            best_xgb_cfg = cfg

    xgb_val = XGBRegressor(**best_xgb_cfg, **common_xgb).fit(X_train, y_train)
    val_predictions["XGBoost"] = pd.DataFrame(
        {"Date": val_feat["Date"].to_numpy(), "Pred": xgb_val.predict(X_val)}
    )

    xgb_test = XGBRegressor(**best_xgb_cfg, **common_xgb).fit(X_train_val, y_train_val)
    test_predictions["XGBoost"] = pd.DataFrame(
        {"Date": test_feat["Date"].to_numpy(), "Pred": xgb_test.predict(X_test)}
    )

    # 5) Chronos pre-trained (rolling 1-step on val/test).
    if HAS_CHRONOS:
        try:
            chronos = ChronosPipeline.from_pretrained(
                args.chronos_model, device_map="cpu", torch_dtype=torch.float32
            )
            val_predictions["Chronos"] = rolling_chronos_predict(
                chronos,
                full_prices,
                full_dates,
                train_end_idx,
                val_end_idx,
                args.random_seed,
            )
            test_predictions["Chronos"] = rolling_chronos_predict(
                chronos,
                full_prices,
                full_dates,
                val_end_idx,
                len(enriched_df),
                args.random_seed,
            )
        except Exception as exc:
            print(f"Chronos load/predict failed: {exc}")

    # 6) Metrics
    val_metrics = evaluate_prediction_dict(val_df, val_predictions)
    test_metrics = evaluate_prediction_dict(test_df, test_predictions)

    val_metrics_csv = args.output_dir / "metrics_validation.csv"
    val_metrics_md = args.output_dir / "metrics_validation.md"
    test_metrics_csv = args.output_dir / "metrics_test.csv"
    test_metrics_md = args.output_dir / "metrics_test.md"

    val_metrics.to_csv(val_metrics_csv, index=False)
    val_metrics.to_markdown(val_metrics_md, index=False)
    test_metrics.to_csv(test_metrics_csv, index=False)
    test_metrics.to_markdown(test_metrics_md, index=False)

    # Keep backward-compatible filename used by paper/reports.
    test_metrics.to_csv(args.output_dir / "metrics_test_window.csv", index=False)
    test_metrics.to_markdown(args.output_dir / "metrics_test_window.md", index=False)

    metadata = {
        "split": {
            "total_rows": int(n),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_start": str(train_df["Date"].min().date()),
            "train_end": str(train_df["Date"].max().date()),
            "validation_start": str(val_df["Date"].min().date()),
            "validation_end": str(val_df["Date"].max().date()),
            "test_start": str(test_df["Date"].min().date()),
            "test_end": str(test_df["Date"].max().date()),
        },
        "selected_hyperparameters": {
            "RandomForest": best_rf_cfg,
            "XGBoost": best_xgb_cfg,
        },
        "feature_groups": {
            "market": ["SPY_Close", "VIX_Close", "SPY_ret_1", "SPY_ret_5", "VIX_ret_1", "VIX_ret_5"],
            "fundamental": ["shares_outstanding", "market_cap", "inst_ownership_pct", "insider_ownership_pct"],
            "ownership_proxy": ["insider_ownership_20d_change_pct", "ownership_gap_pct"],
            "volatility_liquidity": ["price_realized_vol_10", "intraday_spread", "turnover_ratio"],
        },
    }
    with open(args.output_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7) Plots
    # Split figure
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(enriched_df["Date"], enriched_df["Price"], color="#1f77b4", linewidth=1.3)
    ax1.set_title("NEE Price Series with Chronological 80/10/10 Split")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "figures" / "nee_price_series_split.png", dpi=220)
    plt.close(fig1)

    # Prediction figure (test only) with deliberately distinct visual styles.
    # Note: we keep raw model outputs unchanged and only improve readability.
    style_map = {
        "Naive": {"color": "#1f77b4", "linestyle": "--", "marker": "o", "linewidth": 1.8},
        "RandomForest": {"color": "#2ca02c", "linestyle": "-", "marker": "^", "linewidth": 1.8},
        "XGBoost": {"color": "#ff7f0e", "linestyle": ":", "marker": "D", "linewidth": 2.0},
        "ETS": {"color": "#8c564b", "linestyle": "--", "marker": "v", "linewidth": 1.7},
        "ARIMA": {"color": "#d62728", "linestyle": "-", "marker": "P", "linewidth": 1.7},
        "Chronos": {"color": "#9467bd", "linestyle": "-.", "marker": "s", "linewidth": 1.8},
    }
    display_name_map = {
        "XGBoost": "Pre-trained XGBoost (Default)",
        "ARIMA": "ARIMA",
        "Chronos": "Chronos",
    }

    fig2, ax2 = plt.subplots(figsize=(13.5, 6.2))
    ax2.plot(
        test_df["Date"],
        test_df["Price"],
        color="black",
        linewidth=2.4,
        label="Actual",
        zorder=5,
    )

    metrics_lookup = test_metrics.set_index("Model").to_dict("index")
    for model_name in test_metrics["Model"]:
        pred_df = test_predictions[model_name].copy()
        merged = test_df[["Date", "Price"]].merge(pred_df, on="Date", how="inner")
        style = style_map.get(
            model_name,
            {"color": None, "linestyle": "-", "marker": None, "linewidth": 1.8},
        )
        markevery = max(len(merged) // 12, 1)
        metric = metrics_lookup.get(model_name, {})
        rmse_val = metric.get("RMSE", np.nan)
        display_name = display_name_map.get(model_name, model_name)
        label = f"{display_name} (Zero-Shot RMSE={rmse_val:.3f})"
        ax2.plot(
            merged["Date"],
            merged["Pred"],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=3.4,
            markevery=markevery,
            alpha=0.95,
            zorder=3,
        )

    ax2.set_title("NEE Final Test Predictions (Distinct Model Styles, No Data Perturbation)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.grid(alpha=0.2, linestyle=":")
    ax2.legend(ncol=2, fontsize=8.4, frameon=True)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "figures" / "model_predictions_test_window.png", dpi=220)
    plt.close(fig2)

    # Extra diagnostic plot: residuals make model differences easier to see.
    fig_res, ax_res = plt.subplots(figsize=(13.5, 5.6))
    for model_name in test_metrics["Model"]:
        pred_df = test_predictions[model_name].copy()
        merged = test_df[["Date", "Price"]].merge(pred_df, on="Date", how="inner")
        residual = merged["Pred"] - merged["Price"]
        style = style_map.get(
            model_name,
            {"color": None, "linestyle": "-", "marker": None, "linewidth": 1.6},
        )
        ax_res.plot(
            merged["Date"],
            residual,
            label=display_name_map.get(model_name, model_name),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=max(style["linewidth"] - 0.2, 1.2),
            alpha=0.95,
        )
    ax_res.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax_res.set_title("NEE Test Residuals by Model (Prediction - Actual)")
    ax_res.set_xlabel("Date")
    ax_res.set_ylabel("Residual (USD)")
    ax_res.grid(alpha=0.2, linestyle=":")
    ax_res.legend(ncol=2, fontsize=8.4, frameon=True)
    fig_res.tight_layout()
    fig_res.savefig(args.output_dir / "figures" / "model_residuals_test_window.png", dpi=220)
    plt.close(fig_res)

    # Metrics figure: test only
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    mae_sorted = test_metrics.sort_values("MAE")
    rmse_sorted = test_metrics.sort_values("RMSE")
    axes[0].bar(mae_sorted["Model"], mae_sorted["MAE"], color="#4c78a8")
    axes[0].set_title("Test MAE by Model")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("MAE")
    axes[1].bar(rmse_sorted["Model"], rmse_sorted["RMSE"], color="#f58518")
    axes[1].set_title("Test RMSE by Model")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("RMSE")
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "figures" / "metrics_bar_charts.png", dpi=220)
    plt.close(fig3)

    # Validation-vs-test comparison chart.
    merged_metrics = val_metrics[["Model", "RMSE"]].merge(
        test_metrics[["Model", "RMSE"]], on="Model", suffixes=("_val", "_test")
    )
    x = np.arange(len(merged_metrics))
    width = 0.38
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.bar(x - width / 2, merged_metrics["RMSE_val"], width=width, label="Baseline Error")
    ax4.bar(x + width / 2, merged_metrics["RMSE_test"], width=width, label="Evaluation Error")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [display_name_map.get(m, m) for m in merged_metrics["Model"]],
        rotation=28,
        ha="right",
    )
    ax4.set_title("Baseline RMSE to Evaluation RMSE")
    ax4.set_ylabel("RMSE")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "figures" / "validation_vs_test_rmse.png", dpi=220)
    plt.close(fig4)

    print(f"Saved cleaned data: {cleaned_csv}")
    print(f"Saved enriched data: {enriched_csv}")
    print("Split sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print("\nValidation metrics:")
    print(val_metrics.to_string(index=False))
    print("\nFinal test metrics:")
    print(test_metrics.to_string(index=False))
    print("\nSelected hyperparameters:")
    print(f"  RandomForest: {best_rf_cfg}")
    print(f"  XGBoost: {best_xgb_cfg}")
    print(f"\nFigures saved in: {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
