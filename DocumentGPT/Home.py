from datetime import datetime

import streamlit as st

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
