import streamlit as st
import datetime as dt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas as pd
import base64

tickerinput = 'NFLX'

def stock_predict(tickerinput):

    asset_2 = yf.Ticker(tickerinput)
    info = asset_2.info

    history_args = {
        "period": "1y",
        "interval": "1d",
        "start": dt.datetime.now() - dt.timedelta(days=365),
        "end": None,
    }

    p1,periodT, intervalsT,p4 = st.beta_columns(4)
    with p1:
        st.write("")
    with periodT:
        v_T = ['6mo', '1y', '2y', '5y']
        history_args["period"] = st.selectbox(
            "Select Period", options=v_T, index=2
        )
        periodT_2 = history_args["period"]
        if periodT_2 == '6mo':
            v_I = ['1h', '1d']

        if periodT_2 == '1y':
            v_I = ['1h', '1d', '1wk']

        if periodT_2 == '2y':
            v_I = ['1h', '1d', '1wk', '1mo']

        if periodT_2 == '5y':
            v_I = ['1d', '1wk', '1mo', '3mo']

    with intervalsT:
        history_args["interval"] = st.selectbox(
            "Select Interval", options=v_I, index=0
        )
    with p4:
        st.write("")
    intervalT = history_args["interval"]
    periodT = history_args["period"]

    if periodT == '6mo' and intervalT =='1h':
        implies_value = 87
    if periodT == '6mo' and intervalT =='1d':
        implies_value = 13
    if periodT == '1y' and intervalT =='1h':
        implies_value = 176
    if periodT == '1y' and intervalT =='1d':
        implies_value = 26
    if periodT == '1y' and intervalT =='1wk':
        implies_value = 6
    if periodT == '2y' and intervalT =='1h':
        implies_value = 353
    if periodT == '2y' and intervalT =='1d':
        implies_value = 51
    if periodT == '2y' and intervalT =='1wk':
        implies_value = 11
    if periodT == '2y' and intervalT =='1mo':
        implies_value = 3
    if periodT == '5y' and intervalT =='1d':
        implies_value = 126
    if periodT == '5y' and intervalT =='1wk':
        implies_value = 27
    if periodT == '5y' and intervalT =='1mo':
        implies_value = 7
    if periodT == '5y' and intervalT =='3mo':
        implies_value = 3

    ticker_input_2 = yf.Ticker(tickerinput)
    datatest = ticker_input_2.history(period=periodT, interval=intervalT)



    p1,p2,p3 = st.beta_columns([.5,3,.5])
    with p1:
        st.write("")
    with p2:
        st.dataframe(datatest, width=1000)
        fname = st.text_input('Enter here: FILENAME_' + tickerinput + ".csv")

        def download_link(object_to_download, download_filename, download_link_text):

            if isinstance(object_to_download, pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(datatest, fname + '_' + tickerinput + '.csv',
                                              'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    with p3:
        st.write("")

    line_fig = plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Close Prices')
    plt.plot(datatest['Close'])
    plt.title((info['longName']) + ' closing price')

    with st.form(key='figure1'):
        P1_1, P1_2, P1_3 = st.beta_columns([1, 3, 1])
        with P1_1:
            st.write("")
        with P1_2:
            st.subheader("Closing Price")
            st.pyplot(line_fig)
        with P1_3:
            st.write("")
        submit_button = st.form_submit_button(label='Submit')



    df_close = datatest['Close']
    df_close.plot(style='k.')
    plt.title('Scatter plot of closing price')
    scatter_fig = line_fig

    P1_1, P1_2, P1_3 = st.beta_columns([1,3,1])
    with P1_1:
        st.write("")
    with P1_2:
        st.subheader("Scatter Plot : Closing Price")
        st.pyplot(scatter_fig)
    with P1_3:
        st.write("")

    def test_stationarity(timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        # Plot rolling statistics:
        plt.plot(timeseries, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)

        adft = adfuller(timeseries, autolag='AIC')
        # output for dft will give us without defining what the values are.
        # hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],
                           index=['Test Statistics', 'p-value', 'No. of lags used',
                                  'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values

    result = seasonal_decompose(df_close, model='multiplicative', freq=1)
    summary_fig = plt.figure()
    summary_fig = result.plot()
    summary_fig.set_size_inches(16, 9)

    P1_1, P1_2, P1_3 = st.beta_columns([1,3,1])
    with P1_1:
        st.write("")
    with P1_2:
        st.subheader("Seasonal and Trend")
        st.pyplot(summary_fig)
    with P1_3:
        st.write("")


    rcParams['figure.figsize'] = (10, 6)
    df_log = np.log(df_close)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color="black", label="Standard Deviation")
    plt.plot(moving_avg, color="red", label="Mean")
    plt.legend()


    # split data into train and training set
    train_data, test_data = df_log[3:int(len(df_log) * 0.9)], df_log[int(len(df_log) * 0.9):]
    predict_fig = plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()

    P1_1, P1_2, P1_3 = st.beta_columns([1,3,1])
    with P1_1:
        st.write("")
    with P1_2:
        st.subheader("Trained & Test Data")
        st.pyplot(predict_fig)
    with P1_3:
        st.write("")

    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                 test='adf',  # use adftest to find             optimal 'd'
                                 max_p=3, max_q=3,  # maximum p and q
                                 m=1,  # frequency of series
                                 d=None,  # let model determine 'd'
                                 seasonal=False,  # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    fig_5 = model_autoARIMA.plot_diagnostics(figsize=(15, 8))

    P1_1, P1_2, P1_3 = st.beta_columns([1,3,1])
    with P1_1:
        st.write("")
    with P1_2:
        st.subheader("ARIMA Model")
        st.write(fig_5)
    with P1_3:
        st.write("")


    model = ARIMA(train_data, order=(3, 1, 2))
    fitted = model.fit(disp=-1)

    # Forecast
    fc, se, conf = fitted.forecast(implies_value, alpha=0.05)  # 95% confidence
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    fig_6 = plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train_data, label='training')
    plt.plot(test_data, color='blue', label='Actual Stock Price')
    plt.plot(fc_series, color='orange', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    plt.title('Altaba Inc. Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Actual Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # report performance
    mse = mean_squared_error(test_data, fc)

    mae = mean_absolute_error(test_data, fc)

    rmse = math.sqrt(mean_squared_error(test_data, fc))

    mape = np.mean(np.abs(fc - test_data) / np.abs(test_data))

    P1_1, P1_2, P1_3 = st.beta_columns([1,3,1])
    with P1_1:
        st.write("")
    with P1_2:
        st.subheader("Stock Price Prediction")
        st.pyplot(fig_6)
    with P1_3:
        st.write("")


    Fmse = '{0:.3f}'.format(mse)
    st.write('MSE: '+Fmse)

    Fmae = '{0:.3f}'.format(mae)
    st.write('MAE: '+Fmae)

    Frmse = '{0:.3f}'.format(rmse)
    st.write('RMSE: '+Frmse)

    Fmape = '{0:.3f}'.format(mape)
    st.write('MAPE: '+Fmape)

