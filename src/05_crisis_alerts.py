import pandas as pd
import numpy as np
import os

# ── Load files ─────────────────────────────────────────────────
inv     = pd.read_csv('data/predictions/inventory_forecast.csv')
prophet = pd.read_csv('data/predictions/prophet_forecast.csv')
mapping = pd.read_csv('data/dish_ingredient_mapping.csv')

# ── Next week predicted demand per dish ────────────────────────
prophet['ds'] = pd.to_datetime(prophet['ds'])
max_date      = prophet['ds'].max()
next_week     = prophet[prophet['ds'] >= max_date - pd.Timedelta(days=7)]

next_week_demand = (
    next_week.groupby('dish_name')['yhat']
             .sum()
             .reset_index()
)
next_week_demand.columns = ['dish_name', 'predicted_qty_next_week']

# ── Convert dish demand → ingredient burn rate ─────────────────
burn = next_week_demand.merge(
    mapping[['dish_name', 'ingredient_name', 'qty_per_serving_kg']],
    on='dish_name', how='left'
)
burn['predicted_burn_kg'] = (
    burn['predicted_qty_next_week'] * burn['qty_per_serving_kg']
)
weekly_burn = (
    burn.groupby('ingredient_name')['predicted_burn_kg']
        .sum()
        .reset_index()
)

# ── Merge with inventory ───────────────────────────────────────
alerts = inv.merge(weekly_burn, on='ingredient_name', how='left')
alerts['predicted_burn_kg'] = alerts['predicted_burn_kg'].fillna(0)

# ── Shelf life (days) — default 7, update manually if needed ──
shelf_life_map = {
    'Tomatoes': 5, 'Romaine Lettuce': 4, 'Mushrooms': 5,
    'Fresh Basil': 3, 'Milk': 5, 'Cream': 5, 'Eggs': 14,
    'Chicken Breast': 3, 'Beef Patty': 3, 'Lamb Meat': 3,
    'Mango': 5, 'Lemons': 7, 'Avocado': 4, 'Parsley': 4,
}
alerts['shelf_life_days'] = (
    alerts['ingredient_name'].map(shelf_life_map).fillna(14)
)

# ── Day of week demand boost (Friday/Saturday = high volume) ──
import datetime
today     = datetime.datetime.today().weekday()
day_boost = 1.3 if today in [3, 4] else 1.0  # Thu/Fri = boost
alerts['adjusted_burn_kg'] = (
    alerts['predicted_burn_kg'] * day_boost
).round(3)

# ── Crisis alert logic ─────────────────────────────────────────
def get_alert(row):
    stock   = row['closing_stock_kg']
    burn    = row['adjusted_burn_kg']
    reorder = row['reorder_level_kg']
    shelf   = row['shelf_life_days']

    if stock < burn or stock <= reorder:
        return 'RED'
    elif stock > burn * 1.5 and shelf <= 7:
        return 'YELLOW'
    else:
        return 'GREEN'

def get_reason(row):
    stock   = row['closing_stock_kg']
    burn    = row['adjusted_burn_kg']
    reorder = row['reorder_level_kg']
    shelf   = row['shelf_life_days']

    if stock < burn:
        return 'Shortage — stock below predicted weekly burn'
    elif stock <= reorder:
        return 'Below reorder level — place order now'
    elif stock > burn * 1.5 and shelf <= 7:
        return 'Surplus — spoilage risk within shelf life'
    else:
        return 'Inventory aligned with predicted demand'

alerts['alert_status']       = alerts.apply(get_alert,   axis=1)
alerts['alert_reason']       = alerts.apply(get_reason,  axis=1)
alerts['days_of_stock_left'] = (
    (alerts['closing_stock_kg'] /
     alerts['adjusted_burn_kg'].replace(0, np.nan)) * 7
).round(1)

# ── Save ───────────────────────────────────────────────────────
os.makedirs('data/predictions', exist_ok=True)
alerts.to_csv('data/predictions/crisis_alerts.csv', index=False)

# ── Print summary ──────────────────────────────────────────────
print("\n🚨 CRISIS ALERT SUMMARY")
print("=" * 60)
print(alerts.groupby('alert_status')['ingredient_name'].count()
           .rename('count').to_string())
print("\n🔴 RED ALERTS (order immediately):")
red = alerts[alerts['alert_status'] == 'RED']
if len(red):
    print(red[['ingredient_name','closing_stock_kg',
               'adjusted_burn_kg','days_of_stock_left',
               'alert_reason']].to_string(index=False))
else:
    print("  None")

print("\n🟡 YELLOW ALERTS (surplus / spoilage risk):")
yellow = alerts[alerts['alert_status'] == 'YELLOW']
if len(yellow):
    print(yellow[['ingredient_name','closing_stock_kg',
                  'adjusted_burn_kg','shelf_life_days',
                  'alert_reason']].to_string(index=False))
else:
    print("  None")

print("\n✅ crisis_alerts.csv saved —", len(alerts), "ingredients")