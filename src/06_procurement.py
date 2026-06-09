import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Load files ─────────────────────────────────────────────────
alerts = pd.read_csv('data/predictions/crisis_alerts.csv')
inv    = pd.read_csv('data/inventory.csv')

print(f"Alerts loaded   : {len(alerts)} rows")
print(f"Inventory loaded: {len(inv)} rows")

# ── Lead times ─────────────────────────────────────────────────
lead_time_map = {
    'FreshMart'   : 2,
    'GlobalFoods' : 3,
    'LocalFarm'   : 1,
    'QuickSupply' : 2,
    'EcoProvide'  : 3,
    'FarmDirect'  : 1,
}

# ── Only RED and YELLOW ────────────────────────────────────────
to_order = alerts[alerts['alert_status'].isin(['RED', 'YELLOW'])].copy()
print(f"Ingredients to order: {len(to_order)}")

if len(to_order) == 0:
    print("✅ No orders needed — all stock levels are GREEN")
else:
    # ── cost_per_kg_usd already exists in alerts — no merge needed
    # just get supplier from inventory if not already in alerts
    if 'supplier' not in to_order.columns:
        to_order = to_order.merge(
            inv[['ingredient_name', 'supplier']],
            on='ingredient_name', how='left'
        )

    # Fill missing
    to_order['cost_per_kg_usd'] = to_order['cost_per_kg_usd'].fillna(5.0)
    to_order['supplier']        = to_order['supplier'].fillna('Unknown')

    # ── Burn column ────────────────────────────────────────────
    burn_col = 'adjusted_burn_kg' if 'adjusted_burn_kg' in to_order.columns \
               else 'predicted_burn_kg'
    print(f"Using burn column : {burn_col}")

    # ── Order quantity ─────────────────────────────────────────
    to_order['order_qty_kg'] = (
        (to_order[burn_col] * 1.2) - to_order['closing_stock_kg']
    ).clip(lower=0.5).round(2)

    to_order['estimated_cost_usd'] = (
        to_order['order_qty_kg'] * to_order['cost_per_kg_usd']
    ).round(2)

    # ── Dates ──────────────────────────────────────────────────
    def get_delivery(supplier):
        days = lead_time_map.get(str(supplier), 3)
        return (datetime.today() + timedelta(days=days)).strftime('%Y-%m-%d')

    to_order['order_date']        = datetime.today().strftime('%Y-%m-%d')
    to_order['expected_delivery'] = to_order['supplier'].apply(get_delivery)
    to_order['order_status']      = 'Draft'
    to_order['priority']          = to_order['alert_status'].map(
        {'RED': '1 - Urgent', 'YELLOW': '2 - Normal'}
    )

    # ── Select final columns ───────────────────────────────────
    cols = [
        'ingredient_name', 'supplier', 'alert_status', 'priority',
        'closing_stock_kg', burn_col, 'order_qty_kg',
        'estimated_cost_usd', 'order_date', 'expected_delivery',
        'order_status'
    ]
    procurement = to_order[cols].sort_values('priority').reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────
    os.makedirs('data/predictions', exist_ok=True)
    procurement.to_csv('data/predictions/procurement_orders.csv', index=False)

    # ── Print ──────────────────────────────────────────────────
    print("\n🛒 PROCUREMENT ORDERS GENERATED")
    print("=" * 60)
    print(procurement.to_string(index=False))
    print(f"\n💰 Total estimated cost: ${procurement['estimated_cost_usd'].sum():,.2f}")
    print(f"✅ procurement_orders.csv saved — {len(procurement)} orders")