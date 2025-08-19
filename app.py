import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pandas_datareader as data
import yfinance as yf
import seaborn as sns
import streamlit as st
from keras.models import load_model
import plotly.graph_objects as go

st.title("📈Stock Analyzer and Predictor")

stock=st.text_input("Enter Stock Symnbol:", 'GOOG')

start_date = st.date_input("Start Date", pd.to_datetime('2010-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('today'))

model=load_model('Stock Price Prediction.keras')


if st.button("Fetch Data") and stock:
    def fetch_data(ticker):
        try:
            return yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
        
    data = fetch_data(stock)

    if data is not None:
        st.subheader(f"Raw Data for {stock}")
        st.dataframe(data)

        data.dropna(inplace=True)


        st.subheader("Descriptive Statistics")
        st.write(data.describe())

        st.subheader("Fundamental Analysis")
        # --- Get fundamentals ---
        ticker = yf.Ticker(stock)
        ticker_info = ticker.info   # fundamentals

        # --- Build fundamentals dict (raw values only) ---
        fundamentals = {
            "Market Cap": ticker_info.get("marketCap", "N/A"),
            "PE Ratio (TTM)": ticker_info.get("trailingPE", "N/A"),
            "PB Ratio": ticker_info.get("priceToBook", "N/A"),
            "Dividend Yield": ticker_info.get("dividendYield", "N/A"),
            "52-Week High": ticker_info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": ticker_info.get("fiftyTwoWeekLow", "N/A"),
            "Beta": ticker_info.get("beta", "N/A"),
            "EPS (TTM)": ticker_info.get("trailingEps", "N/A"),
            "Revenue (TTM)": ticker_info.get("totalRevenue", "N/A"),
            "Profit Margins": ticker_info.get("profitMargins", "N/A")
        }

        # Convert to DataFrame
        fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"])
        st.table(fundamentals_df)

        st.subheader('Candlesticks Graph (Time Vs Price)')
        data.reset_index(inplace=True)
        data.to_csv("xx.csv")
        data_csv=pd.read_csv("xx.csv")
        data_csv.columns = data_csv.columns.str.strip().str.replace('\ufeff', '')
        fig1 = go.Figure(data=[go.Candlestick(x = data_csv['Date'], open = data_csv['Open'], 
                                    high = data_csv['High'],
                                    low = data_csv['Low'], 
                                    close = data_csv['Close'])])
        fig1.update_layout(width=900,height=500,
                           plot_bgcolor="#192020",   # background of plotting area
                        #    paper_bgcolor="white" ,
                           xaxis_title="Time",yaxis_title="Price",
                           xaxis_rangeslider_visible=False)  # background around plotting area
        st.plotly_chart(fig1)


        #--------------------Closing prices over time----------------------
        st.subheader("Closing Price Over Time")
        # st.line_chart(data['Close'])
        fig11=plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label = f'{stock} Closing Price', linewidth = 1)
        plt.title(f'{stock} Closing prices over time')
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig11)

        #--------------------Opening prices over time----------------------
        st.subheader("Opening prices over time")
        # st.line_chart(data['Open'])
        fig12=plt.figure(figsize=(12, 6))
        plt.plot(data['Open'], label = f'{stock} Opening Price', linewidth = 1)
        plt.title(f'{stock} Opening prices over time')
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig12)

        st.subheader("High Price Over Time")
        # st.line_chart(data['High'])
        fig13=plt.figure(figsize=(12, 6))
        plt.plot(data['High'], label = f'{stock} High Price', linewidth = 1)
        plt.title(f'{stock} High prices over time')
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig13)

        st.subheader("Volume Over Time")
        # st.line_chart(data['Volume'])
        fig14=plt.figure(figsize=(12, 6))
        plt.plot(data['Volume'], label = f'{stock} Volume', linewidth = 2)
        plt.title(f'{stock} Volume over time')
        plt.xlabel('Date/Time Index')
        plt.ylabel('Volume')
        plt.legend()
        st.pyplot(fig14)


        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

        st.subheader("Cumulative Return Over Time")
        st.line_chart(data['Cumulative Return'])

        ma_100=data.Close.rolling(100).mean()
        ma_200=data.Close.rolling(200).mean()

        st.subheader("Moving Averages")
        fig2=plt.figure(figsize=(12, 6))
        plt.plot(data.Close,'g', label = f'{stock} Close Price', linewidth = 1)
        plt.plot(ma_100, 'r',label = f'{stock} Moving Average 100 Price', linewidth = 1)
        plt.plot(ma_200, 'b',label = f'{stock} Moving Average 200 Price', linewidth = 1)
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        st.subheader("Monthly Return Heatmap")
        data.reset_index(inplace=True)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        monthly_return = data['Close'].pct_change().groupby([data['Year'], data['Month']]).mean().unstack()
        fig3=plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_return, cmap='YlGnBu', annot=True, fmt='.2%', cbar=True)
        plt.title('Monthly Return Heatmap')
        st.pyplot(fig3)

        st.subheader("Day of the Week Return Heatmap")
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        weekly_return = data['Close'].pct_change().groupby([data['Year'], data['DayOfWeek']]).mean().unstack()
        fig4=plt.figure(figsize=(12, 4))
        sns.heatmap(weekly_return, cmap='YlOrRd', annot=True, fmt='.2%', cbar=True)
        plt.title('Day of the Week Return Heatmap')
        st.pyplot(fig4)


        # Training & Testing

        data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): int(len(data))])

        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))   

        
        # data_train_sc=sc.fit_transform(data_train)

        # x_train = []
        # y_train = []

        # for i in range(100, data_train_sc.shape[0]):
        #     x_train.append(data_train_sc[i-100:i])
        #     y_train.append(data_train_sc[i, 0])

        # x_train, y_train  = np.array(x_train), np.array(y_train)


        # # Model Building
        # from keras.layers import Dense, Dropout, LSTM
        # from keras.models import Sequential


        # model = Sequential()

        # model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1],1)))
        # model.add(Dropout(0.2))

        # model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
        # model.add(Dropout(0.3))

        # model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
        # model.add(Dropout(0.4))

        # model.add(LSTM(units = 120, activation = 'relu'))
        # model.add(Dropout(0.5))

        # model.add(Dense(units = 1))


        # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        # model.fit(x_train, y_train, epochs = 50)

        past_100_days = data_train.tail(100)
        final_df = pd.concat([past_100_days,data_test], ignore_index = True)
        input_data = sc.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test  = np.array(x_test), np.array(y_test)
        y_test1=y_test     

        y_predicted = model.predict(x_test)

        # a=1/sc.scale_

        # scaler_factor = a
        # y_predicted = y_predicted * scaler_factor
        # y_test = y_test * scaler_factor
        y_predicted = sc.inverse_transform(y_predicted)
        y_test=np.reshape(y_test,(y_test.shape[0],1))
        y_test=sc.inverse_transform(y_test)

        st.subheader("Actual Price Vs Predict Price")

        fig5=plt.figure(figsize=(12, 6))
        plt.plot(y_test,'g', label = 'Original Price', linewidth = 1)
        plt.plot(y_predicted, 'r',label = 'Predicted Price', linewidth = 1)
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        st.pyplot(fig5)




        #Future Stock Price
        m = y_test1
        z= []
        future_days = 10
        for i in range(future_days):
            m = m.reshape(-1,1)
            inter = [m[-100:,0]]
            inter = np.array(inter)
            inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
            pred = model.predict(inter)
            m = np.append(m ,pred)
            z = np.append(z, pred)

        z = np.array(z)
        z = sc.inverse_transform(z.reshape(-1,1))
        st.subheader('Price Prediction of Next 10 Dayes')
        fig6=plt.figure(figsize=(12, 6))
        plt.plot(z, 'r',label = 'Predicted Price', linewidth = 1)
        plt.xlabel('Date/Time Index')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig6)      




