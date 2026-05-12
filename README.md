# Stock Price Volatility Forecasting with Time Series Models

End-to-end pipeline for forecasting stock prices and volatility using classical time series models (ARIMA / SARIMA / Prophet) and a deep learning baseline (LSTM). Ships with feature engineering, evaluation, a long/flat backtester, and an interactive Streamlit dashboard.

## Project layout

```
.
├── config.yaml                # ticker, dates, model hyperparameters
├── requirements.txt
├── src/
│   ├── config.py              # config loader
│   ├── data_loader.py         # Yahoo Finance download + caching
│   ├── feature_engineering.py # SMA, EMA, RSI, Bollinger, MACD, returns
│   ├── volatility.py          # close-to-close, Parkinson, Garman-Klass
│   ├── evaluation.py          # RMSE, MAE, MAPE, direction accuracy
│   ├── backtest.py            # long/flat signal-driven backtest
│   ├── visualization.py       # matplotlib + plotly charts
│   ├── pipeline.py            # end-to-end train/evaluate/backtest
│   └── models/
│       ├── arima_model.py     # ARIMA + SARIMA (statsmodels)
│       ├── prophet_model.py   # Facebook Prophet
│       └── lstm_model.py      # Keras LSTM with walk-forward eval
├── scripts/
│   └── run_pipeline.py        # CLI entry point
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   └── test_basic.py
└── results/                   # generated metrics, plots, saved models
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Prophet wheels can be heavy on Apple Silicon — if it fails to install, the pipeline and dashboard will gracefully skip it.

## Run the full pipeline

```bash
python scripts/run_pipeline.py                       # uses config.yaml
python scripts/run_pipeline.py --ticker MSFT --no-lstm
python scripts/run_pipeline.py --ticker TSLA --start 2019-01-01 --end 2024-12-31
```

Outputs:
- `data/raw/{TICKER}.csv` — raw OHLCV
- `data/processed/{TICKER}_features.csv` — engineered features
- `results/metrics.csv` — model comparison table
- `results/backtest.csv` — equity curve of best model
- `results/plots/*.png` — indicator, forecast, and equity charts
- `results/models/*` — pickled ARIMA/SARIMA/Prophet + Keras LSTM

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard has four tabs:
1. **Overview** — candlestick + volume with moving averages
2. **Indicators** — Bollinger bands, RSI, MACD, realised volatility
3. **Forecasts** — pick models, train, compare against actual + metrics table
4. **Backtest** — long/flat strategy, equity vs buy-and-hold, Sharpe, drawdown

## Configuration

Edit `config.yaml` to change the ticker, date range, train/test split, or model hyperparameters. Key sections:

- `data` — ticker, date range, interval
- `features` — moving-average windows, RSI period, Bollinger band parameters
- `models` — ARIMA `order`, SARIMA `order` + `seasonal_order`, Prophet flags, LSTM lookback/units/epochs
- `backtest` — initial cash and transaction cost

## Models

| Model   | Library     | Strengths                              | Notes                                |
| ------- | ----------- | -------------------------------------- | ------------------------------------ |
| ARIMA   | statsmodels | Simple, interpretable                  | Captures autoregressive structure    |
| SARIMA  | statsmodels | Adds weekly seasonality                | Slower; better for cyclical patterns |
| Prophet | prophet     | Trend + multi-seasonality + holidays   | Handles missing dates gracefully     |
| LSTM    | TensorFlow  | Non-linear, long-range dependencies    | Walk-forward eval over the test set  |

## Evaluation

Each model is scored with **RMSE**, **MAE**, **MAPE**, and **direction accuracy** (fraction of correctly predicted up/down moves). The best RMSE model is automatically passed into the backtester.

## Backtest

Simple long/flat strategy: go long when the forecast's next step is above today's close, otherwise flat. Includes proportional transaction cost. Reports total return, Sharpe ratio, max drawdown, and compares to buy-and-hold.



## Run tests

```bash
pip install pytest
pytest tests/
```

Tests use synthetic data and run fully offline.

## Learning objectives covered

- Time series components (trend, seasonality, noise) — Prophet decomposition + SARIMA
- ARIMA / SARIMA / Prophet implementations
- LSTM for sequence prediction with walk-forward evaluation
- Feature engineering: SMA, EMA, RSI, Bollinger Bands, MACD, OHLC volatility estimators
- Evaluation: RMSE / MAE / MAPE / direction accuracy + backtesting
- Interactive visualisation with Plotly + Streamlit
