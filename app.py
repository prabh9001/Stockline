from sklearn.linear_model import LinearRegression
import numpy as ny
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import yfinance as yf
import mplfinance as mpf
import datetime
#for offline plotting
from plotly.offline import plot

st.title('Stock Price Predicted Line')
# Get the current date

end = datetime.date.today()

# Set the start date to 10 years ago
start = end - datetime.timedelta(days=365*14)#In this case, the code sets the days parameter to 365*14, which calculates the number of days in 10 years (365 days * 10) and multiplies it by 14 to add an additional 4 years as a safety margin to ensure that the data covers a full 10-year period

st.sidebar.write("Note : Enter the Stock Symbol First")
user_input=st.sidebar.text_input('Enter Stock')
st.sidebar.button('Search')
st.header(user_input)

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
    st.write(df.describe())



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



df['Date'] = pd.to_datetime(df['Date'])
# Set 'date_column' as the index
df.set_index('Date', inplace=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Chart With Volume Of Candles')
fig=mpf.plot(df,type='candle',volume=True)
st.pyplot(fig)
#def chart():
  #  df['Date'] = pd.to_datetime(df['Date'])
# Set 'date_column' as the index
#df.set_index('Date', inplace=True)

#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.subheader('(High,Low,Open,Close) Chart And Shows the volume Of candles')
#d=df[['High','Low','Open','Close']]
#fig=mpf.plot(df,type='candle',volume=True)
#st.pyplot(fig)

# Allow the user to enter a date
#user_date1 = st.sidebar.text_input("Enter a date (YYYY-MM-DD):")

# Convert the user-entered date to a datetime object
#user_date1 = pd.to_datetime(user_date1)

#if 'Date' in df.columns:
    #iday = df.loc[df['Date'] == user_date1, :]
    # Plot the intraday data as a candlestick chart with moving averages
    #fig = mpf.plot(iday, type='candle', mav=(7,12))
     # Display the chart using the st.pyplot() function
    #st.pyplot(fig)
#else:
    # Handle the case where the 'Date' column does not exist in the df DataFrame
    #print("The 'Date' column does not exist in the df DataFrame.")



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


st.title('Linear Regression - Actual vs Predicted Values')
from sklearn.linear_model import LinearRegression
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

score=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train,lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train,lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(score)
st.text("Metrics:")
st.text(score)


