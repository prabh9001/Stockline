# Import yfinance
import yfinance as yf
import pandas as pd

def fetch_periods_intervals():
    # Create dictionary for periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods

def fetch_stock_history(user_input, period, interval):
    # Convert the user_input variable to a string
    user_input = str(user_input)

    # Fetch the stock history
    stock_data = yf.Ticker(user_input)

    # Get the stock history for the specified period and interval
    stock_history = stock_data.history(period=period, interval=interval)

    return stock_history