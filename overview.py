from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import requests
import lxml as lh

def Overview(asset):

    ticker = yf.Ticker(asset)
    info = ticker.info

    def candle(asset):
        candle_ex = st.beta_expander("Candlestick Chart Settings", expanded=True)
        with candle_ex:
            intervalList = ["1m", "5m", "15m", "30m"]
            interval_candle = st.selectbox(
                'Interval in minutes',
                intervalList,
            )
            dayList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            chartdays = st.selectbox(
                'No. of Days',
                dayList,
            )
        st.subheader('Market Profile Chart (US S&P 500)')
        stock = yf.Ticker(asset)
        history_data = stock.history(interval=interval_candle, period=str(chartdays) + "d")
        prices = history_data['Close']
        volumes = history_data['Volume']

        lower = prices.min()
        upper = prices.max()

        prices_ax = np.linspace(lower, upper, num=20)

        vol_ax = np.zeros(20)

        for tech_i in range(0, len(volumes)):
            if (prices[tech_i] >= prices_ax[0] and prices[tech_i] < prices_ax[1]):
                vol_ax[0] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[1] and prices[tech_i] < prices_ax[2]):
                vol_ax[1] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[2] and prices[tech_i] < prices_ax[3]):
                vol_ax[2] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[3] and prices[tech_i] < prices_ax[4]):
                vol_ax[3] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[4] and prices[tech_i] < prices_ax[5]):
                vol_ax[4] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[5] and prices[tech_i] < prices_ax[6]):
                vol_ax[5] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[6] and prices[tech_i] < prices_ax[7]):
                vol_ax[6] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[7] and prices[tech_i] < prices_ax[8]):
                vol_ax[7] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[8] and prices[tech_i] < prices_ax[9]):
                vol_ax[8] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[9] and prices[tech_i] < prices_ax[10]):
                vol_ax[9] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[10] and prices[tech_i] < prices_ax[11]):
                vol_ax[10] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[11] and prices[tech_i] < prices_ax[12]):
                vol_ax[11] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[12] and prices[tech_i] < prices_ax[13]):
                vol_ax[12] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[13] and prices[tech_i] < prices_ax[14]):
                vol_ax[13] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[14] and prices[tech_i] < prices_ax[15]):
                vol_ax[14] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[15] and prices[tech_i] < prices_ax[16]):
                vol_ax[15] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[16] and prices[tech_i] < prices_ax[17]):
                vol_ax[16] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[17] and prices[tech_i] < prices_ax[18]):
                vol_ax[17] += volumes[tech_i]

            elif (prices[tech_i] >= prices_ax[18] and prices[tech_i] < prices_ax[19]):
                vol_ax[18] += volumes[tech_i]

            else:
                vol_ax[19] += volumes[tech_i]

        fig_candle = make_subplots(
            rows=1, cols=2,
            column_widths=[0.2, 0.8],
            specs=[[{}, {}]],
            horizontal_spacing=0.01
        )

        fig_candle.add_trace(
            go.Bar(
                x=vol_ax,
                y=prices_ax,
                text=np.around(prices_ax, 2),
                textposition='auto',
                orientation='h'
            ),
            row=1, col=1
        )

        dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")
        fig_candle.add_trace(
            go.Candlestick(x=dateStr,
                           open=history_data['Open'],
                           high=history_data['High'],
                           low=history_data['Low'],
                           close=history_data['Close'],
                           yaxis="y2"

                           ),

            row=1, col=2
        )
        fig_candle.update_layout(
            bargap=0.01,  # gap between bars of adjacent location coordinates,
            showlegend=False,

            xaxis=dict(
                showticklabels=False
            ),
            yaxis=dict(
                showticklabels=False
            ),

            yaxis2=dict(
                title="Price (USD)",
                side="right"
            )
        )
        fig_candle.update_yaxes(nticks=10)
        fig_candle.update_xaxes(nticks=10)
        fig_candle.update_yaxes(side="right")
        fig_candle.update_layout(height=800)
        fig_candle.update_yaxes(gridcolor='#19D3F3')
        fig_candle.update_xaxes(gridcolor='#19D3F3')

        config = {
            'modeBarButtonsToAdd': ['drawline']
        }
        st.plotly_chart(fig_candle, use_container_width=True, config=config)

    candle(asset)
    left, right = st.beta_columns([1, 1])
    with left:
        st.write("WAP")

    with right:
        st.subheader("About")
        st.info(info['longBusinessSummary'])



