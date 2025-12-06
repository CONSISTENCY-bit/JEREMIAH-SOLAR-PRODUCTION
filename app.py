# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import holidays
from datetime import datetime
from prophet import Prophet

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    prophet_model = pickle.load(open("prophet_model.pkl", "rb"))
    residual_model = pickle.load(open("xgb_residual_model.pkl", "rb"))
    return prophet_model, residual_model

prophet_model, residual_model = load_models()

# -------------------------------
# HOLIDAY CHECK FOR GHANA
# -------------------------------
def is_gh_holiday(date):
    gh_holidays = holidays.CountryHoliday("GH")
    return 1 if date in gh_holidays else 0

# -------------------------------
# SEASON ASSIGNMENT
# -------------------------------
def assign_season(month):
    if month in [11, 12, 1]:
        return "harmattan"
    elif month in [6, 7, 8]:
        return "rainy"
    else:
        return "dry"

# -------------------------------
# CYCLIC TIME FEATURES
# -------------------------------
def add_cyclic_features(df):
    day = 60 * 60 * 24
    year = 365.2425 * day
    seconds = (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    df["DaySin"] = np.sin(seconds * (2 * np.pi / day))
    df["DayCos"] = np.cos(seconds * (2 * np.pi / day))
    df["YearSin"] = np.sin(seconds * (2 * np.pi / year))
    df["YearCos"] = np.cos(seconds * (2 * np.pi / year))
    return df

# -------------------------------
# FEATURE ENGINEERING PIPELINE
# -------------------------------
def build_features(date):
    df = pd.DataFrame(index=[pd.to_datetime(date)])

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day_of_Week"] = df.index.dayofweek
    df["Day_of_Year"] = df.index.dayofyear
    df["WeekOfYear"] = df.index.isocalendar().week.astype(int)
    df["Hour"] = df.index.hour
    df["Quarter"] = df.index.quarter

    # Ghana holiday flag
    df["gh_holidays"] = df.index.map(is_gh_holiday)

    # Season encoding
    df["season"] = df["Month"].apply(assign_season)
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

    # Add cyclic features
    df = add_cyclic_features(df)

    return df

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Hybrid Solar Forecast", layout="centered")
st.title("🌞 Hybrid Solar Production Predictor")
st.markdown("This app uses **Prophet + XGBoost** to forecast solar energy production in Ghana. Select a date and time to get a prediction.")

# Date and time input
selected_date = st.date_input("📅 Select Date")
selected_time = st.time_input("⏰ Select Time")
chosen_datetime = datetime.combine(selected_date, selected_time)

# Predict button
if st.button("🔍 Predict"):
    # Prophet forecast
    prophet_input = pd.DataFrame({"ds": [chosen_datetime]})
    prophet_pred = prophet_model.predict(prophet_input)["yhat"].iloc[0]

    # Feature engineering
    features = build_features(chosen_datetime)

    # Residual prediction with XGBoost
    residual_pred = residual_model.predict(features)[0]

    # Final hybrid prediction
    final_output = prophet_pred + residual_pred

    st.success(f"🌤️ Predicted Solar Production: **{final_output:.2f} kWh**")
