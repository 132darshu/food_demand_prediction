Food Demand Prediction & Smart Inventory Dashboard

An AI-powered system that predicts restaurant dish demand and automates inventory management to reduce food waste and prevent stockouts. Built as part of PS-II internship at KPTAC Technologies, Dubai.

Project Overview
- Forecasts dish demand using Prophet and XGBoost
- Generates RED, YELLOW, GREEN inventory crisis alerts automatically
- Auto-creates draft procurement orders with quantities and costs
- Analyses food waste by dish and branch
- Interactive dashboard built with Streamlit

Dataset
- 1,000 restaurant orders across 1 full year
- 25 dishes, 5 branches, 81 ingredients
- 6 interconnected datasets

 Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas and NumPy | Data processing |
| Prophet | Time series forecasting |
| XGBoost | Demand prediction |
| Streamlit | Web dashboard |
| Plotly | Charts |
| Power BI | Business reporting |

How to Run
Install libraries first:
```bash
pip install -r requirements.txt
```
Then run scripts in order:
```bash
python src/01_data_prep.py
python src/02_prophet.py
python src/03_xgboost.py
python src/04_inventory.py
python src/05_crisis_alerts.py
python src/06_procurement.py
streamlit run app/app.py
```

Dashboard Screenshots

Overview
<img width="767" height="434" alt="overview" src="https://github.com/user-attachments/assets/481650f6-bba6-48aa-a084-3f58df74d23d" />
<img width="755" height="350" alt="overview2" src="https://github.com/user-attachments/assets/c011c782-53a4-4c4c-a8c3-52c645279c12" />


Demand Forecast
<img width="742" height="373" alt="demand forcast" src="https://github.com/user-attachments/assets/8e9aee57-e9f8-4567-9fac-42ba23e89589" />
<img width="724" height="313" alt="demand forcast2" src="https://github.com/user-attachments/assets/835ecb51-1609-485a-87d8-8927ac35e14b" />

Inventory Alerts
<img width="761" height="383" alt="inventory" src="https://github.com/user-attachments/assets/07dca41a-28ef-45ea-a808-6c796b2b004d" />

Procurement Orders
<img width="748" height="392" alt="Procurement" src="https://github.com/user-attachments/assets/703dda6d-36e3-489a-a9f4-075d1e95808c" />

Waste Analysis
<img width="753" height="409" alt="Food Waste Analysis" src="https://github.com/user-attachments/assets/1f14fd21-4ffa-44e8-9869-50159beb49f6" />

## Project Structure
```
food_demand_prediction/
├── data/
├── src/
├── app/
├── screenshots/
└── requirements.txt
```
