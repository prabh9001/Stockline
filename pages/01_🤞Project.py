import streamlit as st

from sklearn.linear_model import LinearRegression
import numpy as ny
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objs as go

from plotly.offline import plot
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import datetime
#for offline plotting
from plotly.offline import plot

from timeinterval import fetch_periods_intervals, fetch_stock_history

st.set_page_config(
    page_title="Prediction",
    page_icon="📊",
)

st.title('Stock Price Predicted Line')
# Get the current date

end = datetime.date.today()

# Set the start date to 10 years ago
start = end - datetime.timedelta(days=365*14)
#In this case, the code sets the days parameter to 365*14, which calculates the number of days in 10 years (365 days * 10) and multiplies it by 14 to add an additional 4 years as a safety margin to ensure that the data covers a full 10-year period

st.sidebar.write("Note : Enter the Stock ticker Symbol First")
user_input=st.sidebar.text_input('Enter Stock ticker')
st.sidebar.button('Search')
st.header(user_input)

def get_stock_history(symbol):
    stock_data = yf.download(symbol)
    return stock_data

def get_stock_name(symbol):
    stock = yf.Ticker(symbol)
    return stock.info['shortName']
if user_input:
    stock_name = get_stock_name(user_input)
    st.write(f"Stock Name: {stock_name}")


df = yf.Ticker(user_input)
Cashflow_Statement,Balance_Sheet,Holders_Info=st.tabs(["Cashflow Statement Of the company","Balance Sheet Of the company","Insider Roster Holders Info"])
with Holders_Info:
    df.insider_roster_holders
with Cashflow_Statement:
    df.cashflow
with Balance_Sheet:
    df.balance_sheet

Data_Chart,major_holders, Dividend_and_stock_split=st.tabs(["Data Chart","Major Holders in the company","Shows the Dividend and stock split given the company"])
with major_holders:
    df.major_holders
with Dividend_and_stock_split:
    df.actions
   

df = yf.download(user_input, start, end)
st.sidebar.write("Data Starting date :",start)
st.sidebar.write("Data Ending date :",end)


 #Describing data
with Data_Chart:
    st.subheader('Data Chart')
    df = df.reset_index()
    st.write(df)

#Visualisation
st.subheader('Closing Price Chart with Date ')
layout = go.Layout(
    title='Stock price',
    xaxis=dict(
        title='Year',
        titlefont=dict(
            family='Courier New,monospace',
            size=18,
            color='red'
        )
    ),
yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New,monospace',
            size=18,
            color='red'
        )
        )
        )
Stock_data=[{'x':df['Date'],'y':df['Close']}]
plot= go.Figure(data=Stock_data,layout=layout)
st.plotly_chart(plot)

#for Date error in predicted chart
df['Date'] = pd.to_datetime(df['Date'])
# Set 'date_column' as the index
df.set_index('Date', inplace=True)

st.header("Stock Intraday Data")
# Get the user input for the period
periods = fetch_periods_intervals()

# Add a selector for period
period = st.selectbox("Choose a period", list(periods.keys()))

# Add a selector for interval
interval = st.selectbox("Choose an interval", periods[period])

# Fetch the stock history if the user inputs are valid
if user_input and period and interval:
    stock_history = fetch_stock_history(user_input, period, interval)
    # Create the plotly figure


# Create a new trace for the candlestick chart
candlestick = go.Candlestick(
    x=stock_history.index,
    open=stock_history['Open'],
    high=stock_history['High'],
    low=stock_history['Low'],
    close=stock_history['Close']
)

# Create the plotly figure
figure = go.Figure(
    data=[
        candlestick
    ],
    layout=go.Layout(
        title=f'{stock_name} Intraday Chart',
        xaxis=go.layout.XAxis(title='Date'),
        yaxis=go.layout.YAxis(title='Value')
    )
)

# Display the plot in Streamlit
st.plotly_chart(figure)

One_Hundred_Days_EMA_Chart,Two_Hundred_Days_EMA_Chart=st.tabs(["100 Days EMA Chart","200 Days EMA Chart"])
with One_Hundred_Days_EMA_Chart:
    #using 100 days EMA Indicator for analysis
    st.subheader('100 Days EMA Chart')
    ma100= df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100)
    st.pyplot(fig)
with  Two_Hundred_Days_EMA_Chart:
    #using 200 days EMA Indicator for analysis
    st.subheader('200 Days EMA Chart')
    ma100= df.Close.rolling(100).mean()
    ma200= df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100)
    plt.plot(ma200)
    st.pyplot(fig)


def LinearRegression():
    from sklearn.linear_model import LinearRegression
    
    st.title('Linear Regression - Actual vs Predicted Values')
# Example data
X_train = ny.random.rand(2274, 1)  # Replace with your actual features
y_train = ny.random.rand(2274)

#now we can split the data into training set and testing set
from numpy import reshape

X=ny.array(df.index).reshape(-1, 1)
Y=df['Close']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
scaler= StandardScaler().fit(X_train)

# Create and fit a linear model
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Check and convert data types if needed
if X_train.dtype == ny.dtype('datetime64[ns]'):
    # Convert datetime to numerical representation
    X_train = X_train.astype(ny.float64)

# Convert X_train to a Pandas dataframe
X_train = pd.DataFrame(X_train)
# Convert the first column of X_train to a datetime object
X_train['Date'] = pd.to_datetime(X_train.iloc[:, 0])
# Set the first column of X_train as the index
X_train.set_index('Date', inplace=True)

trace0=go.Scatter(
    x=X_train.index,
    y=Y_train,
    mode='markers',
    name= 'Actual'
    )
trace1=go.Scatter(
    x=X_train.index,
    y=lm.predict(X_train).T,
    mode='lines',
    name= 'Predicted'
    )
data=[trace0,trace1]

# Set the xaxis type to 'date'
layout.xaxis.type = 'date'
layout.xaxis.title.text = 'Date'
plot2=go.Figure(data=data, layout=layout)
st.plotly_chart(plot2)
#Cal score of final model evaluation

#for model evaulation and for accuracy
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
# Check and convert data types if needed
if X_test.dtype == ny.dtype('datetime64[ns]'):
    # Convert datetime to numerical representation
    X_test = X_test.astype(ny.float64)

st.text("This Metric Shows the trained data and testing data :")
score=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train,lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train,lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(score)
st.text("Metrics:")
st.text(score)


