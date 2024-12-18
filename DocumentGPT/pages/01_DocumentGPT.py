import time

import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🤖",
)

st.title("Document GPT")

# human 이 작성한 메세지
with st.chat_message("human"):
    st.write("Hellooooo!")

with st.chat_message("ai"):
    st.write("How are you!!!!!")

st.chat_input("Send a message to the ai")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="Error", state="error")
