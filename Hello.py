# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Write the following in two separate terminals:
# Step 1: pip install -r requirements.txt
# Step 2: streamlit run /workspaces/model-sim/app.py

# Import libraries
import streamlit as st
import pandas as pd
#import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from plotly import graph_objs as go
#from scipy.stats import sem, t
#import plotly.express as px
from streamlit.logger import get_logger
#from prophet import Prophet
#from prophet.plot import plot_plotly

LOGGER = get_logger(__name__)

# Variables
#Input parameter values
days = 30
future_days = 5
simulations = 1000

# Page configuration
st.set_page_config(
    page_title="Stock Price Dashboard",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded")


def describe_data():
  st.subheader('Price Data')

# Data configuration, formatting
full_data = pd.read_csv('data/SP500.csv')
df = full_data.copy()
df['Date'] = pd.to_datetime(df['DATE'])
# Replace '.' with NaN and forward fill the missing values
df['Close'] = df['SP500'].replace('.', pd.NA).ffill()

# Convert 'Close' column to float
df['Close'] = df['Close'].astype(float)

#######################
# Functions
# Calculates the mean and 95% confidence interval highs and lows of the 30 days before each date
def calculate_stats(df):
    # Initialize the new columns with NaN values
    df['mean_close'] = np.nan
    df['ci_high'] = np.nan
    df['ci_low'] = np.nan
    
    # Iterate over the DataFrame rows
    for index in range(30, len(df)):
        # Get the current date
        current_date = df.iloc[index]['Date']
        
        # Get last 30 days as a range
        start_date = current_date - timedelta(days=30)
        period_df = df[(df['Date'] >= start_date) & (df['Date'] < current_date)]
        
        # Mean, Standard Deviation
        mean_close = period_df['Close'].mean()
        std_close = period_df['Close'].std()
        
        # 95% CI calculations
        z_score = 1.96
        margin_error = z_score * std_close / np.sqrt(len(period_df))
        ci_high = mean_close + margin_error
        ci_low = mean_close - margin_error
        
        # Truncate values to two decimal places and update the DataFrame
        df.at[index, 'mean_close'] = round(mean_close, 2)
        df.at[index, 'ci_high'] = round(ci_high, 2)
        df.at[index, 'ci_low'] = round(ci_low, 2)
    
    # Return the DataFrame with the new columns
    return df

# Pre-run:
# This is to initialize the dataframe with extra columns for mean, 95% CI high, 95% CI low
df_with_stats = calculate_stats(df)


def plot_raw_data():
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],
                  y=df['Close'], name="Closing Price", line=dict(color='firebrick', width=4)))
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['mean_close'], name="30-day Mean", line=dict(color='blue', width=2)))

   
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ci_high'], name="95% CI High", line=dict(color='green', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ci_low'], name="95% CI Low", line=dict(color='green', width=2, dash='dash')))

   
    yaxis_buttons = [
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")
    ]

    # Adjusts Y-axis
    fig.update_layout(
        title='S&P 500 Price with Mean, 95% CI',
        xaxis=dict(
            rangeselector=dict(
                buttons=yaxis_buttons
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        yaxis=dict(
            title='Closing Price',
            rangemode='tozero'
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"yaxis.range": [4800, 5400]}],
                        label="1m",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.range": [4250, 5500]}],
                        label="YTD",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.range": [3500, 5500]}],
                        label="1y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.range": [1500, 5500]}],
                        label="All",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.2,
                xanchor="left",
                y=1.25,
                yanchor="top"
            ),
        ]
    )
    fig.show()


# Predicts the prices using a Monte Carlo simulation, adjustable parameters
def predict_stock_prices(df, days=30, simulations=1000, future_days=5):
    # Calculate daily returns (% relative to past X days)
    daily_returns = df['Close'].iloc[-days:].pct_change().dropna()

    # Calculate the mean and standard deviation of daily returns
    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()

    # Set up the Monte Carlo simulation
    results = np.zeros((simulations, future_days))

    # Run simulations
    for i in range(simulations):
        # Generates simulated daily returns based on mean, SD
        simulated_returns = np.random.normal(mean_return, std_dev_return, future_days)
        # Calculate the price path (cumulative product)
        price_path = df['Close'].iloc[-1] * (1 + simulated_returns).cumprod()
        results[i, :] = price_path

    return results

# Generates a histogram of likely price ranges
def plot_simulation_results(simulation_results):
    # Takes the last day prices from each simulation
    last_day_prices = simulation_results[:, -1]
    fig = go.Figure(data=[go.Histogram(x=last_day_prices, nbinsx=50)])
    # Displays updated chart in the GUI
    fig.update_layout(
        title_text=f"Frequency of {simulations} Predicted Prices over the next {future_days} days",
        xaxis_title="Price",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig)
    
    
#######################
# Sidebar
with st.sidebar:
    #st.title('🔢 Stock Price Dashboard')
    
    drop_list = st.sidebar.selectbox("SPX", ('Price Depiction', 'Price Prediction'), 0)
    
    
    
try:
    if drop_list == 'Price Depiction':
        #test
        st.subheader("S&P 500 Time-Series Analysis")
        
        df_with_stats = calculate_stats(df)
        plot_raw_data()
    
    if drop_list == 'Price Prediction':
        #Test
        st.subheader("S&P 500 Monte Carlo Prediction")
        days = int(st.text_input('Days', 30))
        future_days = int(st.text_input('Future Days', 5))
        simulations = int(st.text_input('Simulation Count', 1000))
        
        simulation_results = predict_stock_prices(df, days, simulations, future_days)
        plot_simulation_results(simulation_results)
except ValueError:
    st.error('Symbol cannot be empty !')

except ConnectionError:
    st.error('Could not connect to the internet :(')

#def run():
#  st.write("# Welcome to Streamlit! 👋")



if __name__ == "__main__":
    df_with_stats = calculate_stats(df)

