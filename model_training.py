import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# Load Data
df = pd.read_csv("Walmart.csv")

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Features
X = df[
    [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Week"
    ]
]

# Target
y = df["Weekly_Sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("RMSE :", rmse)
print("MAE  :", mae)
print("R2   :", r2)

# Save Model
joblib.dump(model, "xgb_model.pkl")

print("Model Saved Successfully")