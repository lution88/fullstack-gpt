import os
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
from openai import OpenAIError


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def folder_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def validate_api_key(api_key):
    """API 키의 유효성을 검사합니다."""
    try:
        # 테스트용 LLM 인스턴스 생성
        test_llm = ChatOpenAI(temperature=0.1, openai_api_key=api_key)
        # 간단한 테스트 메시지로 API 키 검증
        test_llm.predict("테스트 메시지입니다.")
        return True, None
    except OpenAIError as e:
        return False, str(e)
    except Exception as e:
        return False, f"알 수 없는 오류가 발생했습니다: {str(e)}"


def load_openai_api_key():
    """Streamlit 사이드바에서 OpenAI API 키를 로드하고 설정합니다."""
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False

    if "api_key_error" not in st.session_state:
        st.session_state.api_key_error = None

    with st.sidebar:
        st.markdown("## OpenAI API 설정")
        api_key = st.text_input(
            "OpenAI API 키를 입력하세요:",
            type="password",
            help="OpenAI API 키는 안전하게 저장되며 세션 내에서만 사용됩니다.",
            key="openai_api_key",
        )

        if api_key:
            # API 키 유효성 검사
            is_valid, error_message = validate_api_key(api_key)

            if is_valid:
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.api_key_configured = True
                st.session_state.api_key_error = None
                st.success("✅ API 키가 성공적으로 설정되었습니다!")
                return True
            else:
                st.session_state.api_key_configured = False
                st.session_state.api_key_error = error_message
                st.error(f"⚠️ API 키 오류: {error_message}")
                return False
        else:
            st.session_state.api_key_configured = False
            st.warning("⚠️ API 키를 입력해주세요.")
            return False


def handle_chat_error(func):
    """채팅 관련 함수의 예외를 처리하는 데코레이터"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OpenAIError as e:
            st.error(f"OpenAI API 오류: {str(e)}")
            # API 키 상태 초기화
            st.session_state.api_key_configured = False
            st.warning("API 키를 다시 확인해주세요.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

    return wrapper


@handle_chat_error
def process_message(message, chain):
    """메시지 처리 및 응답 생성"""
    with st.chat_message("ai"):
        chain.invoke(message)


st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    try:
        file_content = file.read()
        file_folder = "./.cache/files"
        folder_exist(file_folder)
        file_path = f"{file_folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        cache_folder = "./.cache/embeddings"
        folder_exist(cache_folder)
        cache_dir = LocalFileStore(f"{cache_folder}/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except OpenAIError as e:
        st.error(f"임베딩 생성 중 OpenAI API 오류: {str(e)}")
        return None
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {str(e)}")
        return None


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

st.title("Quiz GPT")

st.markdown(
    """
Welcome!!!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

# API 키 확인
is_api_key_configured = load_openai_api_key()

if is_api_key_configured:
    try:
        # API 키가 설정된 경우에만 LLM 초기화
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )

        with st.sidebar:
            file = st.file_uploader(
                "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
            )

        if file:
            retriever = embed_file(file)
            if retriever is not None:  # 임베딩 성공한 경우에만 진행
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
                    process_message(message, chain)
        else:
            st.session_state["messages"] = []

    except OpenAIError as e:
        st.error(f"OpenAI API 오류: {str(e)}")
        st.session_state.api_key_configured = False
        st.warning("API 키를 다시 확인해주세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
else:
    st.info("👈 사이드바에서 OpenAI API 키를 먼저 설정해주세요.")
