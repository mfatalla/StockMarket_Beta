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
    st.write("WAP")

    left, right = st.beta_columns([1, 1])
    with left:
        st.write("")

    with right:
        st.subheader("About")
        st.info(info['longBusinessSummary'])



