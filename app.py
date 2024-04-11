# pip install streamlit streamlit-chat langchain openai tiktoker faiss-cpu
# Sample prompts: Tell me about the headphones in your store

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import os

OPENAI_API_KEY = "ADD YOUR API KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

with st.spinner("Fetching Data"):
    if "vectors" not in st.session_state:
        loader = CSVLoader(file_path="products.csv", encoding="utf-8")
        data = loader.load()
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        st.session_state["vectors"] = vectors
st.success("Chatbot Ready")


def conversation(query):
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"),
        retriever=st.session_state["vectors"].as_retriever(),
    )
    result = chain({"question": query, "chat_history": st.session_state["history"]})
    st.session_state["history"].append((query, result["answer"]))
    return result["answer"]


if "history" not in st.session_state:
    st.session_state["history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hello, Tell me your product requirements and I will suggest you a product"
    ]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hey"]


# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_input(
            "Query:", placeholder="What product are you looking for", key="input"
        )
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = conversation(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="big-smile",
            )
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
