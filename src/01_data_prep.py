import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

print("📂 Working from:", os.getcwd())

# ── Load datasets ──────────────────────────────────────────────
orders    = pd.read_csv('data/restaurant_orders.csv')
weekly    = pd.read_csv('data/weekly_demand_summary.csv')
customers = pd.read_csv('data/customers.csv')

print(f"Orders loaded   : {len(orders)} rows")
print(f"Weekly loaded   : {len(weekly)} rows")
print(f"Customers loaded: {len(customers)} rows")

# ── One row per order ──────────────────────────────────────────
orders_clean = orders.drop_duplicates(subset='order_id').copy()
print(f"After dedup     : {len(orders_clean)} unique orders")

orders_clean['order_date'] = pd.to_datetime(orders_clean['order_date'])
orders_clean['month']      = orders_clean['order_date'].dt.month
orders_clean['dayofweek']  = orders_clean['order_date'].dt.dayofweek

# ── Merge customer info ────────────────────────────────────────
ml_df = orders_clean.merge(
    customers[['customer_id', 'age_group', 'loyalty_tier']],
    on='customer_id', how='left'
)
print(f"After merge     : {len(ml_df)} rows")

# ── Encode categoricals ────────────────────────────────────────
cat_cols = ['dish_category', 'day_of_week', 'season',
            'weather_condition', 'dine_in_or_takeaway',
            'promotion_applied', 'restaurant_branch']

for col in cat_cols:
    if col in ml_df.columns:
        ml_df[col + '_enc'] = pd.factorize(ml_df[col])[0]
        print(f"  Encoded: {col}_enc")
    else:
        print(f"  ⚠️ Column not found: {col}")

# ── Feature engineering ────────────────────────────────────────
ml_df = ml_df.sort_values('order_date')
ml_df['lag_1week_qty'] = (
    ml_df.groupby('dish_name')['quantity_ordered'].shift(7)
)
ml_df['rolling_7day_avg'] = (
    ml_df.groupby('dish_name')['quantity_ordered']
         .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# ── Save ───────────────────────────────────────────────────────
ml_df.to_csv('data/ml_ready.csv', index=False)
print(f"\n✅ ml_ready.csv saved — {len(ml_df)} rows")
print("Columns:", list(ml_df.columns))