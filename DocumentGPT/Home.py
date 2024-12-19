from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="FullstackGPT HOME",
    page_icon="📃",
)

st.markdown(
    """

# Hello!

Welcome to my FullstackGPT Portfolio!

Here are the apps I made:

- [x] [DocumentGPT](/DocumentGPT)
- [ ] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

model = st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
    ),
)

st.write(model)


name = st.text_input("What is your name")
st.write(name)

value = st.slider(
    "temperature",
    min_value=0.1,
    max_value=1.0,
)
st.write(value)

# 사이드바
st.sidebar.title("sidebar title")
st.sidebar.text_input("xxx")

# 사이드바 - with keyword
with st.sidebar:
    st.title("hello")
    st.text_input("sidebar with 'with keyword'")

tab_1, tab_2, tab_3 = st.tabs(["a", "b", "c"])
with tab_1:
    st.write("a")

with tab_2:
    st.write("b")

with tab_3:
    st.write("c")
