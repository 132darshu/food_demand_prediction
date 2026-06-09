import pandas as pd
import numpy as np
import os

orders    = pd.read_csv('data/restaurant_orders.csv')
inventory = pd.read_csv('data/inventory.csv')

daily_usage = (
    orders.groupby(['order_date', 'ingredient_name'])['ingredient_quantity_used_kg']
          .sum().reset_index()
)

avg_daily = (
    daily_usage.groupby('ingredient_name')['ingredient_quantity_used_kg']
               .mean().reset_index()
)
avg_daily.columns = ['ingredient_name', 'avg_daily_usage_kg']

inv_forecast = inventory.merge(avg_daily, on='ingredient_name', how='left')
inv_forecast['days_of_stock_left'] = (
    inv_forecast['closing_stock_kg'] /
    inv_forecast['avg_daily_usage_kg'].replace(0, np.nan)
).round(1)

def recommend(row):
    if row['reorder_triggered'] == 'Yes':
        return 'Order NOW'
    elif pd.notna(row['days_of_stock_left']) and row['days_of_stock_left'] < 7:
        return 'Order this week'
    else:
        return 'Stock OK'

inv_forecast['recommendation'] = inv_forecast.apply(recommend, axis=1)

os.makedirs('data/predictions', exist_ok=True)
inv_forecast.to_csv('data/predictions/inventory_forecast.csv', index=False)
print("✅ Inventory forecast saved")
print(inv_forecast[['ingredient_name', 'closing_stock_kg',
                     'days_of_stock_left', 'recommendation']].to_string())