import time

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        '''llm이 언제 토큰을 생성하기 시작하는지
        - llm이 새 토큰을 생성하기 시작하면, 화면에 empty box를 생성한다.
        '''
        # with st.sidebar:
        #     st.write("llm started!")
        self.message_box = st.empty()  # 빈 위젯

    def on_llm_end(self, *args, **kwargs):
        """llm이 말하기를 멈추었을 때, 작업을 끝냈을 때 호출
        - llm이 언제 작업을 끝내는지
        이 때 self.message에는 전체 메세지가 들어있다.
        """
        save_message(self.message, "ai")
        # with st.sidebar:
        #     st.write("llm ended!")

    def on_llm_new_token(self, token, *args, **kwargs):
        """chain을 invoke할때 호출
        - llm이 언제 new token을 생성하는지
        - new tkoen을 받으면 해당 token을 메세지에 추가한다.
        """
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# @st.cache_data - streamlit은 파일 확인용.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 경로
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # retriever의 임무는 단지 너에게 documents를 제공하는 것. chain에서 쓸 수 있도록!!
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# template 만들기 -> prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context.
            If you don't know the answer just say you don't know.
            Don't make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Document GPT")

# 1. 시작은 사용자에게 파일을 upload할 것을 요청.
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file..")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        # with 블록 내부 chain invoke ->
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []
