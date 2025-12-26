import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from simfin_api import SimFinAPI
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path

def get_simfin_api_key() -> str | None:
    # 1) Streamlit Cloud / deployment secrets
    try:
        if "SIMFIN_API_KEY" in st.secrets:
            return st.secrets["SIMFIN_API_KEY"]
    except Exception:
        pass

    # 2) Local dev: keys.env file
    if Path("keys.env").exists():
        load_dotenv("keys.env")
        key = os.getenv("SIMFIN_API_KEY")
        if key:
            return key

    # 3) Public/demo fallback: allow manual entry
    st.sidebar.markdown("### 🔑 SimFin API Key")
    return st.sidebar.text_input(
        "Enter your SimFin API key",
        type="password",
        help="Used only for this session to fetch live data. Not stored."
    )

api_key = get_simfin_api_key()

if not api_key:
    st.warning("Please add a SimFin API key via Streamlit Secrets, keys.env, or the sidebar input to continue.")
    st.stop()

api = SimFinAPI(api_key=api_key)

# Configure the page layout
st.set_page_config(page_title="Stock Market Live Analysis", layout="wide")

# Sidebar stock selection
st.sidebar.title("📊 Select a Stock")
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
selected_stock = st.sidebar.radio("Choose a stock:", stocks)

# Page title
st.title(f"📈 Live Trading - {selected_stock}")

# Set time range
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# Adjust end_date based on the weekday
today = datetime.today()
weekday = today.weekday()
if weekday == 0:  # Monday → Use last Friday's data
    end_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
elif weekday == 6:  # Sunday → Use last Friday's data
    end_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
else:  # Normal case: Use yesterday's data
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

# Fetch stock price data
st.write(f"📡 Fetching {selected_stock} stock data from SimFin API... Please wait.")
try:
    share_prices_df = api.get_share_prices(selected_stock, start_date, end_date)
    income_df = api.get_income_statement(selected_stock, start_date, end_date)
    balance_sheet_df = api.get_balance_sheet(selected_stock, start_date, end_date)
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

shares_outstanding_df = api.get_shares_outstanding(selected_stock, start_date, end_date)


# Convert date columns to datetime format
share_prices_df["date"] = pd.to_datetime(share_prices_df["date"])
income_df["date"] = pd.to_datetime(income_df["date"])
balance_sheet_df["date"] = pd.to_datetime(balance_sheet_df["date"])
shares_outstanding_df["date"] = pd.to_datetime(shares_outstanding_df["date"])


# Merge datasets
merged_df = share_prices_df.merge(income_df, on=["ticker", "date"], how="left")
merged_df = merged_df.merge(balance_sheet_df, on=["ticker", "date"], how="left")
merged_df = merged_df.merge(shares_outstanding_df, on=["ticker", "date"], how="left")


# Sort and forward-fill missing values
merged_df = merged_df.sort_values(by=["ticker", "date"], ascending=[True, True])
merged_df.ffill(inplace=True)

# Compute P/E ratio
merged_df["market_capitalization"] = merged_df["close"] * merged_df["shares_outstanding"]
merged_df["p_e_ratio"] = merged_df["market_capitalization"] / merged_df["net_income"]

# Compute 50-day SMA
merged_df["sma_50"] = merged_df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=50, min_periods=1).mean())

# Add next day's close price as a target variable
merged_df["next_close"] = merged_df.groupby("ticker")["close"].shift(-1)

# Drop rows where critical features contain NaN values
merged_df = merged_df.dropna(subset=["close", "p_e_ratio", "sma_50"])

# Drop the fiscal_period column if it exists
if "fiscal_period" in merged_df.columns:
    merged_df = merged_df.drop(columns=["fiscal_period"])


# Display stock data
st.subheader(f"📊 Historical Data for {selected_stock}")
st.dataframe(merged_df)

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("mag7_final_model.json")

# Ensure date is normalized (no time component surprises)
merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.normalize()

requested_date = pd.to_datetime(end_date).normalize()

# Pick the latest available date <= requested_date
available_dates = merged_df.loc[merged_df["date"] <= requested_date, "date"]
if available_dates.empty:
    st.warning("⚠️ No data available on or before the requested end_date.")
    st.stop()

prediction_date = available_dates.max()

yesterday_df = merged_df.loc[merged_df["date"] == prediction_date, ["ticker", "close", "p_e_ratio", "sma_50"]]
st.caption(f"Using latest available trading day for prediction: {prediction_date.date()}")

if not yesterday_df.empty:
    # Make a prediction using the model
    dmatrix = xgb.DMatrix(yesterday_df[["close", "p_e_ratio", "sma_50"]])
    prediction = model.predict(dmatrix)[0]
    
    # Determine buy/sell signals
    prediction_label = "📈 Buy" if prediction > 0.5 else "📉 Sell"
    yesterday_df["Prediction"] = prediction_label
    
    # Display predictions
    st.subheader("📊 Prediction for Today's Close Price Movement")
    st.write(f"🔮 **{prediction_label}** signal for {selected_stock}")
    st.dataframe(yesterday_df)
else:
    st.warning("⚠️ No available stock data for predictions.")

# Plot Closing Price Trend
st.subheader(f"📈 Closing Price Trend for {selected_stock} (Last Year)")
plt.figure(figsize=(10, 5))
plt.plot(share_prices_df["date"], share_prices_df["close"], label="Closing Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title(f"{selected_stock} Closing Price Over the Last Year")
plt.legend()
st.pyplot(plt)
