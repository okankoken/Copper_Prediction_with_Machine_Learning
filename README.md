# Copper Prediction with Machine Learning

Monthly copper price forecasting project focused on the LME Copper Cash Settlement price.

The project combines market, macroeconomic, mining, shipping, equity, risk, and energy-transition data to forecast copper prices using classical time-series models and machine-learning models.

## Project Goals

- Forecast monthly LME copper prices
- Produce a 12-month forward forecast curve
- Compare short-, medium-, and long-horizon performance
- Evaluate both price accuracy and directional accuracy
- Track experiments with MLflow
- Automate ingestion, quality checks, feature engineering, and forecasting with Airflow

## Forecast Target

Target:

`cash_settlement_usd_per_ton`

Frequency:

`Monthly`

Production horizon:

`H1-H12`

Current benchmark horizons:

- H1
- H3
- H6
- H12

## Models

Current model families:

- Naive / Random Walk
- ARIMA
- SARIMAX
- Ridge Regression
- Elastic Net
- LightGBM
- XGBoost

Additional planned components:

- Return-target tree models
- Direction classification
- Per-horizon model selection
- Ensemble forecasting

## Project Structure

```text
config/
data/
docker/
src/
  ingestion/
  quality/
  features/
  models/
  utils/
airflow/
notebooks/
