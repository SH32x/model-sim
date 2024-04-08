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
input_date = '2024-04-05'
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
# Define a method to calculate the mean and 95% confidence interval highs and lows of the 30 days before each date
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
# Calculate the stats and add them to the DataFrame
df_with_stats = calculate_stats(df)

# Define a method to get the closing price for a given date
def get_closing_price(input_date):
    # Convert input_date to datetime
    input_date = pd.to_datetime(input_date)
    
    # Get the previous Friday's closing price
    previous_friday = input_date - timedelta(days=(input_date.weekday() + 2) % 7)

    # If the input date is a weekend, adjust to the previous Friday
    if input_date.weekday() > 4:  # 5: Saturday, 6: Sunday
        input_date -= timedelta(days=input_date.weekday() - 4)
    
    # Get the row with the closest date before or equal to the input date
    row = df[df['Date'] <= input_date].iloc[-1]

    
    previous_friday_row = df[df['Date'] <= previous_friday].iloc[-1]
    
    # Return the closing price and the previous Friday's closing price
    return row['Close'], previous_friday_row['Close']

# Example usage:
# Assuming the user wants the closing price for '2024-03-16' (which is a Saturday)
#input_date = '2024-03-29'
#closing_price, previous_friday_price = get_closing_price(input_date)
#print(f"The closing price for {input_date} is {closing_price}")
#print(f"The closing price for the previous Friday is {previous_friday_price}")

def plot_raw_data():
    # Plotting the raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],
                  y=df['Close'], name="Closing Price", line=dict(color='firebrick', width=4)))
    # Plotting the 30-day mean
    fig.add_trace(go.Scatter(x=df['Date'], y=df['mean_close'], name="30-day Mean", line=dict(color='blue', width=2)))

    # Plotting the confidence intervals
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ci_high'], name="95% CI High", line=dict(color='green', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ci_low'], name="95% CI Low", line=dict(color='green', width=2, dash='dash')))

    # Define buttons for custom range selection
    yaxis_buttons = [
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")
    ]

    # Update layout to include y-axis range buttons
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
# plot_raw_data()

# Predicts the prices using a Monte Carlo simulation
def predict_stock_prices(df, days=30, simulations=1000, future_days=5):
    # Calculate daily returns
    daily_returns = df['Close'].iloc[-days:].pct_change().dropna()

    # Calculate the mean and standard deviation of daily returns
    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()

    # Set up the Monte Carlo simulation
    results = np.zeros((simulations, future_days))

    # Run simulations
    for i in range(simulations):
        # Generate simulated daily returns
        simulated_returns = np.random.normal(mean_return, std_dev_return, future_days)
        # Calculate the price path
        price_path = df['Close'].iloc[-1] * (1 + simulated_returns).cumprod()
        results[i, :] = price_path

    return results

# Generates a histogram of likely price ranges
def plot_simulation_results(simulation_results):
    # Take the last day prices from each simulation
    last_day_prices = simulation_results[:, -1]

    fig = go.Figure(data=[go.Histogram(x=last_day_prices, nbinsx=50)])
    fig.update_layout(
        title_text=f"Frequency of {simulations} Predicted Prices over the next {future_days} days",
        xaxis_title="Price",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig)
    
    # Plot a histogram of the last day prices
    #plt.hist(last_day_prices, bins=50, alpha=0.75)
    #plt.title(f"Frequency of {simulations} Predicted Prices over the next {future_days} days")
    #plt.xlabel('Price')
    #plt.ylabel('Frequency')
    #plt.show()

# simulation_results = predict_stock_prices(df, days, simulations, future_days)
# plot_simulation_results(simulation_results)
#######################
# Sidebar
with st.sidebar:
    #st.title('🔢 Stock Price Dashboard')
    
    drop_list = st.sidebar.selectbox("SPX", ('Price Depiction', 'Price Prediction'), 0)
    
    
    
try:
    if drop_list == 'Price Depiction':
        #test
        st.subheader("S&P 500 Time-Series Analysis")
        

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



#if __name__ == "__main__":
#    run()

