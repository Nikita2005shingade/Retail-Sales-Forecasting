import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv("Walmart.csv")

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("xgb_model.pkl")

# -----------------------------
# Title
# -----------------------------

st.set_page_config(
    page_title="Retail Sales Forecasting",
    layout="wide"
)

st.title("📈 Retail Sales Forecasting Dashboard")

st.markdown(
    "Forecast future retail sales using Machine Learning"
)

# -----------------------------
# KPIs
# -----------------------------

total_sales = df["Weekly_Sales"].sum()
avg_sales = df["Weekly_Sales"].mean()
stores = df["Store"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "Average Sales",
    f"${avg_sales:,.0f}"
)

col3.metric(
    "Stores",
    stores
)

# -----------------------------
# Sales Trend
# -----------------------------

st.subheader("Weekly Sales Trend")

trend = df.groupby(
    "Date"
)["Weekly_Sales"].sum().reset_index()

fig = px.line(
    trend,
    x="Date",
    y="Weekly_Sales",
    title="Sales Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Monthly Sales
# -----------------------------

df["Month"] = df["Date"].dt.month

monthly = df.groupby(
    "Month"
)["Weekly_Sales"].mean().reset_index()

fig2 = px.bar(
    monthly,
    x="Month",
    y="Weekly_Sales",
    title="Average Monthly Sales"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------------
# Top Stores
# -----------------------------

top = (
    df.groupby("Store")
    ["Weekly_Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

top = top.reset_index()

fig3 = px.bar(
    top,
    x="Store",
    y="Weekly_Sales",
    title="Top 10 Stores"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# -----------------------------
# Prediction Section
# -----------------------------

st.header("Predict Weekly Sales")

store = st.number_input(
    "Store",
    min_value=1,
    max_value=45,
    value=1
)

holiday = st.selectbox(
    "Holiday Week",
    [0,1]
)

temp = st.number_input(
    "Temperature",
    value=70.0
)

fuel = st.number_input(
    "Fuel Price",
    value=3.0
)

cpi = st.number_input(
    "CPI",
    value=200.0
)

unemployment = st.number_input(
    "Unemployment",
    value=8.0
)

year = st.number_input(
    "Year",
    value=2013
)

month = st.slider(
    "Month",
    1,
    12,
    1
)

week = st.slider(
    "Week",
    1,
    52,
    1
)

if st.button("Predict Sales"):

    input_data = pd.DataFrame(
        [[
            store,
            holiday,
            temp,
            fuel,
            cpi,
            unemployment,
            year,
            month,
            week
        ]],
        columns=[
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
    )

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"Predicted Weekly Sales: ${prediction:,.2f}"
    )