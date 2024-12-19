from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="homework 6",
    page_icon="📃",
)


today = datetime.today().strftime("%H:%M:%S")
st.title(today)

# 사이드바 - with keyword
with st.sidebar:
    st.title("OpenAI API KEY")
    st.text_input("Insert OpenAI-api-key")
