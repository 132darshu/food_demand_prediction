import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ── Load ───────────────────────────────────────────────────────
df = pd.read_csv('data/ml_ready.csv')
print("Loaded ml_ready.csv —", len(df), "rows")
print("Null counts:\n", df.isnull().sum())

# ── Features ───────────────────────────────────────────────────
features = ['month', 'dayofweek', 'peak_hour_flag', 'price_per_item',
            'promotion_applied', 'dish_category_enc', 'season_enc',
            'weather_condition_enc', 'dine_in_or_takeaway_enc',
            'restaurant_branch_enc']

# ── Check which feature columns actually exist ─────────────────
missing_cols = [f for f in features if f not in df.columns]
if missing_cols:
    print("⚠️ Missing columns:", missing_cols)
    features = [f for f in features if f in df.columns]
    print("Using these features instead:", features)

# ── Encode promotion_applied if not already numeric ───────────
if df['promotion_applied'].dtype == object:
    df['promotion_applied'] = (df['promotion_applied'] == 'Yes').astype(int)

# ── Fill nulls instead of dropping rows ───────────────────────
X = df[features].fillna(0)
y = df['quantity_ordered']

print(f"\nFinal dataset: {len(X)} rows, {len(features)} features")
print("Sample X:\n", X.head(3))

# ── Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── Train model ────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nMAE : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R²  : {r2_score(y_test, y_pred):.3f}")

# ── Save predictions ───────────────────────────────────────────
os.makedirs('data/predictions', exist_ok=True)
results = X_test.copy()
results['actual_qty']    = y_test.values
results['predicted_qty'] = y_pred.round(1)
results.to_csv('data/predictions/xgboost_predictions.csv', index=False)

# ── Save model ─────────────────────────────────────────────────
with open('src/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ XGBoost model saved + predictions exported")