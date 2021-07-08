import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import Ticker
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def Scrappy(tickerinput):
    def calcMovingAverage(datatech, size):
        dftech = datatech.copy()
        dftech['sma'] = dftech['Adj Close'].rolling(size).mean()
        dftech['ema'] = dftech['Adj Close'].ewm(span=size, min_periods=size).mean()
        dftech.dropna(inplace=True)
        return dftech


    def calc_macd(datatech):
        dftech = datatech.copy()
        dftech['ema12'] = dftech['Adj Close'].ewm(span=12, min_periods=12).mean()
        dftech['ema26'] = dftech['Adj Close'].ewm(span=26, min_periods=26).mean()
        dftech['macd'] = dftech['ema12'] - dftech['ema26']
        dftech['signal'] = dftech['macd'].ewm(span=9, min_periods=9).mean()
        dftech.dropna(inplace=True)
        return dftech


    def calcBollinger(datatech, size):
        dftech = datatech.copy()
        dftech["sma"] = dftech['Adj Close'].rolling(size).mean()
        dftech["bolu"] = dftech["sma"] + 2 * dftech['Adj Close'].rolling(size).std(ddof=0)
        dftech["bold"] = dftech["sma"] - 2 * dftech['Adj Close'].rolling(size).std(ddof=0)
        dftech["width"] = dftech["bolu"] - dftech["bold"]
        dftech.dropna(inplace=True)
        return dftech


    st.title('Technical Indicators')
    st.subheader('Moving Average')

    coMA1, coMA2 = st.beta_columns(2)

    with coMA1:
        numYearMA_list1 = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        query_params = st.experimental_get_query_params()
        default = int(query_params["numYearMA"][0]) if "numYearMA" in query_params else 1
        numYearMA = st.selectbox(
            "Insert period (Year): ",
            numYearMA_list1,
            index=default
        )

    with coMA2:
        windowSizeMA_list2 = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]
        query_params3 = st.experimental_get_query_params()
        default = int(query_params3["windowSizeMA"][0]) if "windowSizeMA" in query_params3 else 19
        windowSizeMA = st.selectbox(
            "Window Size (Day): ",
            windowSizeMA_list2,
            index=default
        )

    start_tech = dt.datetime.today() - dt.timedelta(numYearMA * 365)
    end_tech = dt.datetime.today()
    dataMA = yf.download(tickerinput, start_tech, end_tech)
    df_ma = calcMovingAverage(dataMA, windowSizeMA)
    df_ma = df_ma.reset_index()

    figMA = go.Figure()

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['Adj Close'],
            name="Prices Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['sma'],
            name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['ema'],
            name="EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    figMA.update_layout(legend_title_text='Trend')
    figMA.update_yaxes(tickprefix="$")

    st.plotly_chart(figMA, use_container_width=True)

    st.subheader('Moving Average Convergence Divergence (MACD)')

    numYearMACD_list1 = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    query_params2 = st.experimental_get_query_params()
    default = int(query_params2["numYearMACD"][0]) if "numYearMACD" in query_params2 else 1
    numYearMACD = st.selectbox(
        " Insert period (Year):  ",
        numYearMACD_list1,
        index=default
    )

    startMACD = dt.datetime.today() - dt.timedelta(numYearMACD * 365)
    endMACD = dt.datetime.today()
    dataMACD = yf.download(tickerinput, startMACD, endMACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()

    figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['Adj Close'],
            name="Prices Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema12'],
            name="EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema26'],
            name="EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['macd'],
            name="MACD Line"
        ),
        row=2, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['signal'],
            name="Signal Line"
        ),
        row=2, col=1
    )

    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))

    figMACD.update_yaxes(tickprefix="$")
    st.plotly_chart(figMACD, use_container_width=True)

    st.subheader('Bollinger Band')
    coBoll1, coBoll2 = st.beta_columns(2)

    with coBoll1:
        numYearBoll_list1 = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        query_params4 = st.experimental_get_query_params()
        default = int(query_params4["numYearBoll"][0]) if "numYearBoll" in query_params4 else 1
        numYearBoll = st.selectbox(
            "Insert period (Year) ",
            numYearBoll_list1,
            index=default
        )

    with coBoll2:
        windowSizeBoll_list2 = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]
        query_params4 = st.experimental_get_query_params()
        default = int(query_params4["windowSizeBoll"][0]) if "windowSizeBoll" in query_params4 else 19
        windowSizeBoll= st.selectbox(
            "Window Size (Day) ",
            windowSizeBoll_list2,
            index=default
        )


    startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(tickerinput, startBoll, endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bolu'],
            name="Upper Band"
        )
    )
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['sma'],
            name="SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
        )
    )
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bold'],
            name="Lower Band"
        )
    )
    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)

    st.subheader("Simple Moving Average Chart")
    linechart = st.beta_container()
    with linechart:
        linechart_expander = st.beta_expander(label='Line Chart Settings')
        with linechart_expander:
            ticker = yf.Ticker(asset)
            info = ticker.info
            attri = ['SMA', 'SMA2']
            attributes = st.multiselect(
                'Choose Chart Attributes [SMA, SMA2]',
                attri,
                default='SMA'
            )
            data0 = load_quotes(asset)
            data = data0.copy().dropna()
            data.index.name = None
            section = st.slider(
                "Number of quotes",
                min_value=30,
                max_value=min([2000, data.shape[0]]),
                value=500,
                step=10,
            )
            data2 = data[-section:]["Adj Close"].to_frame("Adj Close")
            if "SMA" in attributes:
                period = st.slider(
                    "SMA period", min_value=5, max_value=500, value=20, step=1
                )
                data[f"SMA {period}"] = data["Adj Close"].rolling(period).mean()
                data2[f"SMA {period}"] = data[f"SMA {period}"].reindex(data2.index)
            if "SMA2" in attributes:
                period2 = st.slider(
                    "SMA2 period", min_value=5, max_value=500, value=100, step=1
                )
                data[f"SMA2 {period2}"] = data["Adj Close"].rolling(period2).mean()
                data2[f"SMA2 {period2}"] = data[f"SMA2 {period2}"].reindex(data2.index)
        st.subheader("Chart")
        st.line_chart(data2, height=700)
        if st.checkbox("View quotes"):
            st.subheader(f"{asset} historical data")
            st.write(data2)

    tech_2 = st.beta_container

    predict_chart = st.beta_container()
    with predict_chart:

        history_args = {
            "period": "1y",
            "interval": "1d",
            "start": dt.datetime.now() - dt.timedelta(days=365),
            "end": None,
        }

        history_args["period"] = st.selectbox(
            "Select Period", options=Ticker.PERIODS, index=5  # pylint: disable=protected-access
        )
        history_args["interval"] = st.selectbox(
            "Select Interval", options=Ticker.INTERVALS, index=8  # pylint: disable=protected-access
        )
        intervalT = history_args["interval"]
        periodT = history_args["period"]

        ticker_input_2 = yf.Ticker(asset)
        datatest = ticker_input_2.history(period=periodT, interval=intervalT)

        line_fig = plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        plt.plot(datatest['Close'])
        plt.title((asset) + ' closing price')

        st.subheader("Figure1")
        st.pyplot(line_fig)

        df_close = datatest['Close']
        df_close.plot(style='k.')
        plt.title('Scatter plot of closing price')
        st.subheader("Figure 2")
        scatter_fig = line_fig
        st.pyplot(scatter_fig)

        # Test for staionarity
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
                               index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
            for key, values in adft[4].items():
                output['critical value (%s)' % key] = values

        result = seasonal_decompose(df_close, model='multiplicative', freq=30)
        summary_fig = plt.figure()
        summary_fig = result.plot()
        summary_fig.set_size_inches(16, 9)

        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(df_close)
        moving_avg = df_log.rolling(12).mean()
        std_dev = df_log.rolling(12).std()
        plt.legend(loc='best')
        plt.title('Moving Average')
        plt.plot(std_dev, color="black", label="Standard Deviation")
        plt.plot(moving_avg, color="red", label="Mean")
        plt.legend()
        st.subheader("Figure 3")
        st.pyplot(summary_fig)

        # split data into train and training set
        train_data, test_data = df_log[3:int(len(df_log) * 0.9)], df_log[int(len(df_log) * 0.9):]
        predict_fig = plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(df_log, 'green', label='Train data')
        plt.plot(test_data, 'blue', label='Test data')
        plt.legend()
        st.subheader("Figure 4")
        st.pyplot(predict_fig)
        st.write(len(df_log))

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

        st.write(fig_5)

        model = ARIMA(train_data, order=(3, 1, 2))
        fitted = model.fit(disp=-1)

        # Forecast
        fc, se, conf = fitted.forecast(1, alpha=0.05)  # 95% confidence
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)
        wap = plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(train_data, label='training')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(fc_series, color='orange', label='Predicted Stock Price')
        plt.fill_between(lower_series.index, lower_series, upper_series,
                         color='k', alpha=.10)
        plt.title('Altaba Inc. Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        st.pyplot(wap)

        # report performance
        mse = mean_squared_error(test_data, fc)
        st.write('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data, fc)
        st.write('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        st.write('RMSE: ' + str(rmse))
        mape = np.mean(np.abs(fc - test_data) / np.abs(test_data))
        st.write('MAPE: ' + str(mape))

