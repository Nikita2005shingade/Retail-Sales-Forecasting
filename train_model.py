import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Step 1")

df = pd.read_csv("Walmart.csv")

print("Step 2")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

print("Step 3")

os.makedirs("outputs", exist_ok=True)

sales = df.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(10,5))
plt.plot(sales.index, sales.values)
plt.savefig("outputs/sales_trend.png")
plt.close()

print("Sales Trend Saved")

plt.figure(figsize=(6,4))
sns.barplot(x="Holiday_Flag", y="Weekly_Sales", data=df)
plt.savefig("outputs/holiday_effect.png")
plt.close()

print("Holiday Effect Saved")

print("Finished")

# Monthly Sales

df["Month"] = df["Date"].dt.month

monthly = df.groupby(
    "Month"
)["Weekly_Sales"].mean()

plt.figure(figsize=(8,5))

monthly.plot(kind="bar")

plt.title("Monthly Sales")

plt.savefig(
    "outputs/monthly_sales.png"
)

plt.close()

print("Monthly Sales Saved")
top = df.groupby(
    "Store"
)["Weekly_Sales"].sum()

top = top.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(8,5))

top.plot(kind="bar")

plt.title("Top 10 Stores")

plt.savefig(
    "outputs/top_stores.png"
)

plt.close()

print("Top Stores Saved")

plt.figure(figsize=(8,5))

sns.heatmap(
    df[
        [
            "Weekly_Sales",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment"
        ]
    ].corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.savefig(
    "outputs/heatmap.png"
)

plt.close()

print("Heatmap Saved")