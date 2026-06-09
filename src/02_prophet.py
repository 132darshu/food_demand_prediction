import pandas as pd
from prophet import Prophet
import os

weekly = pd.read_csv('data/weekly_demand_summary.csv')
os.makedirs('data/predictions', exist_ok=True)

all_forecasts = []

for dish in weekly['dish_name'].unique():
    df_dish = weekly[weekly['dish_name'] == dish][['week_number', 'total_qty']].copy()
    df_dish.columns = ['ds', 'y']
    df_dish['ds'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
        (df_dish['ds'] - 1) * 7, unit='D'
    )

    if len(df_dish) < 5:
        continue

    model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_dish)

    future   = model.make_future_dataframe(periods=4, freq='W')
    forecast = model.predict(future)
    forecast['dish_name'] = dish
    all_forecasts.append(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'dish_name']]
    )

result = pd.concat(all_forecasts)
result.to_csv('data/predictions/prophet_forecast.csv', index=False)
print("✅ Prophet forecast saved —", len(result), "rows")