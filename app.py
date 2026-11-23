import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ensemble Learning Dashboard", layout="wide")

# Larger tab labels
st.markdown(
    """
    <style>
    button[data-baseweb="tab"] p {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    preds = pd.read_csv("test_predictions.csv", parse_dates=["target_date"])
    with open("models/feature_list.json") as f:
        features = json.load(f)
    return results, preds, features


@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
    }


results, preds, FEATURES = load_data()
models = load_models()
MODEL_COLS = ["Random Forest", "XGBoost", "ARIMA", "Stacking", "Hybrid"]

st.title("Ensemble Learning Dashboard: Colombo Maximum Temperature")

tab_overview, tab_compare, tab_predictions, tab_live = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Predictions",
        "Feature importance & live prediction",
    ]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Project overview")
    st.markdown(
        "- **Data:** daily weather for Colombo, 2010-01-01 to 2023-06-17 (4,916 days)\n"
        "- **Target:** tomorrow's maximum temperature (°C)\n"
        "- **Split:** training 2010–2021, testing 2022–2023\n"
        "- **Models:** Random Forest (bagging), XGBoost (boosting), "
        "Stacking (RF + ARIMA with a Ridge meta-learner), Hybrid (ARIMA + Random Forest on residuals)"
    )
    best = (
        results[results["model"] != "Naive (tomorrow = today)"]
        .sort_values("RMSE")
        .iloc[0]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model (lowest RMSE)", best["model"])
    c2.metric("Best RMSE", f"{best['RMSE']:.3f} °C")
    c3.metric("Best R²", f"{results['R2'].max():.3f}")

# ---------- Model comparison ----------
with tab_compare:
    st.subheader("Model comparison on the test set")
    st.dataframe(results.sort_values("RMSE").round(3), hide_index=True)

    metric = st.selectbox("Metric to chart", ["RMSE", "MAE", "R2"])
    ascending = metric != "R2"
    chart = results.set_index("model")[[metric]].sort_values(
        metric, ascending=ascending
    )
    st.bar_chart(chart)
    st.caption("Lower is better for RMSE and MAE. Higher is better for R².")

# ---------- Predictions ----------
with tab_predictions:
    st.subheader("Actual vs predicted maximum temperature")
    chosen = st.multiselect("Models to show", MODEL_COLS, default=["XGBoost", "Hybrid"])
    d_min = preds["target_date"].min().date()
    d_max = preds["target_date"].max().date()
    date_range = st.date_input(
        "Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max
    )

    if len(date_range) == 2 and chosen:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        view = preds[(preds["target_date"] >= start) & (preds["target_date"] <= end)]

        st.line_chart(view.set_index("target_date")[["actual"] + chosen])

        rmse_range = np.sqrt(((view[chosen].sub(view["actual"], axis=0)) ** 2).mean())
        st.markdown("**RMSE for the selected date range**")
        st.dataframe(rmse_range.rename("RMSE").round(3).to_frame())
    else:
        st.info("Select at least one model and a full date range.")

# ---------- Feature importance and live prediction ----------
with tab_live:
    st.subheader("Feature importance")
    c1, c2 = st.columns(2)
    for col, name in zip([c1, c2], ["Random Forest", "XGBoost"]):
        imp = pd.Series(models[name].feature_importances_, index=FEATURES).sort_values(
            ascending=False
        )
        col.markdown(f"**{name}**")
        col.bar_chart(imp)

    st.subheader("Live prediction")
    st.caption(
        "Enter today's weather and recent history. The models predict the next day's maximum temperature."
    )
    with st.form("live_prediction"):
        c1, c2, c3 = st.columns(3)
        inputs = {
            "tmax_today": c1.number_input("Today's max temp (°C)", value=30.0),
            "tmax_lag1": c1.number_input("Max temp 1 day ago (°C)", value=30.0),
            "tmax_lag2": c1.number_input("Max temp 2 days ago (°C)", value=30.0),
            "tmax_lag3": c1.number_input("Max temp 3 days ago (°C)", value=30.0),
            "tmax_lag7": c2.number_input("Max temp 7 days ago (°C)", value=30.0),
            "temperature_2m_min": c2.number_input("Today's min temp (°C)", value=24.0),
            "temperature_2m_mean": c2.number_input(
                "Today's mean temp (°C)", value=26.5
            ),
            "precipitation_sum": c2.number_input(
                "Today's rainfall (mm)", value=0.0, min_value=0.0
            ),
            "precipitation_hours": c2.number_input(
                "Hours of rain today", value=0.0, min_value=0.0, max_value=24.0
            ),
            "precip_7d_mean": c3.number_input(
                "7-day average rainfall (mm)", value=7.0, min_value=0.0
            ),
            "windspeed_10m_max": c3.number_input(
                "Max wind speed (km/h)", value=14.0, min_value=0.0
            ),
            "winddirection_10m_dominant": c3.number_input(
                "Dominant wind direction (°)",
                value=180.0,
                min_value=0.0,
                max_value=360.0,
            ),
            "et0_fao_evapotranspiration": c3.number_input(
                "Evapotranspiration (mm)", value=3.7, min_value=0.0
            ),
        }
        day = c3.date_input("Date of today", value=pd.Timestamp("2023-06-01").date())
        submitted = st.form_submit_button("Predict tomorrow's maximum temperature")

    if submitted:
        doy = pd.Timestamp(day).dayofyear
        inputs["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        inputs["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        X_new = pd.DataFrame([inputs])[FEATURES]

        c1, c2 = st.columns(2)
        c1.metric(
            "Random Forest", f"{models['Random Forest'].predict(X_new)[0]:.2f} °C"
        )
        c2.metric("XGBoost", f"{models['XGBoost'].predict(X_new)[0]:.2f} °C")
        st.caption(
            "Stacking and the hybrid need ARIMA's recent history, so they appear in the comparison and prediction charts only."
        )
