import streamlit as st
import pandas as pd

def About():
    st.write("")
    stock_forecast = st.beta_expander("Stock Market Forecast", expanded=False)
    with stock_forecast:
        st.write("Slapsoil Stock Market Forecast")
        q1, q2, q3 = st.beta_columns([1.5, 1, 1])
        with q1:
            st.write("")
        with q3:
            st.write("")
        with q2:
            st.image('data//logo1.png')
    st.write("")
    developer_ex = st.beta_expander("Developer", expanded=False)
    with developer_ex:
        st.subheader("Team Slapsoil")
        st.write("")
        st.write("")
        dev_email = "contact.email@dlsud.edu.ph"
        img1, img2, img3, img4, img5, img6, img7 = st.beta_columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
        with img2:
            st.image('data//developer1.png')
            st.write("")
            st.write("")
            st.write("")
            dev1 = "CO"
            dev11 = 'Paolo Henry'
            st.markdown(
                f"<p style='text-align: center;font-weight: bold; color: #FFFFFF;font-size: 30px;'>{dev1}</p>",
                unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center; color: #FFFFFF;font-size: 20px;'>{dev11}</p>",
                unsafe_allow_html=True)
            dev_email = "(" + dev_email + ")"
            aye = '[contact.email@dlsud.edu.ph]' + dev_email
            st.markdown(aye, unsafe_allow_html=True)
        with img4:
            st.image('data//developer1.png')
            st.write("")
            st.write("")
            st.write("")
            dev2 = "FATALLA"
            dev21 = 'Mark'
            st.markdown(
                f"<p style='text-align: center;font-weight: bold; color: #FFFFFF;font-size: 30px;'>{dev2}</p>",
                unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center; color: #FFFFFF;font-size: 20px;'>{dev21}</p>",
                unsafe_allow_html=True)
            dev_email = "(" + dev_email + ")"
            aye = '[contact.email@dlsud.edu.ph]' + dev_email
            st.markdown(aye, unsafe_allow_html=True)

        with img6:
            st.image('data//developer1.png')
            st.write("")
            st.write("")
            st.write("")
            dev3 = "GUTIERREZ"
            dev31 = 'Kenn Carlo'
            st.markdown(
                f"<p style='text-align: center;font-weight: bold; color: #FFFFFF;font-size: 30px;'>{dev3}</p>",
                unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center; color: #FFFFFF;font-size: 20px;'>{dev31}</p>",
                unsafe_allow_html=True)
            dev_email = "(" + dev_email + ")"
            aye = '[contact.email@dlsud.edu.ph]' + dev_email
            st.markdown(aye, unsafe_allow_html=True)

        with img1:
            st.write("")
        with img3:
            st.write("")
        with img5:
            st.write("")
        with img7:
            st.write("")
    st.write("")
    project_depend = st.beta_expander("Project Dependencies", expanded=False)
    with project_depend:
        first_col = ['streamlit', 'pandas', 'request', 'bs4', 'beautifulsoup4', 'lmxl', 'yfinance', 'plotly', 'numpy']
        second_col = ['0.84.0', '1.2.4', '2.25.1', '0.0.1', '4.9.3', '4.6.3', '0.1.59', '4.14.3', '1.20.2']
        requirements = pd.DataFrame(
            {"Dependencies": list(first_col), "Version": list(second_col)})
        requirements.index = [""] * len(requirements)
        st.subheader("Requirements")
        st.table(requirements)
    st.write("")
    git_hub = st.beta_expander("Git Hub", expanded=False)
    with git_hub:
        git_hub_link2 = "https: // github.com / mfatalla / StockMarket"
        git_hub_link2 = "(" + git_hub_link2 + ")"
        git_hub_link_p = "[https: // github.com / mfatalla / StockMarket]" + git_hub_link2
        st.markdown(git_hub_link_p, unsafe_allow_html=True)