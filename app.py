import glob
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load env variables
load_dotenv()
st.set_page_config(page_title="🪙 Crypto AI Assistant", layout="centered")
st.markdown("# 🪙 CryptoGPT — Your Smart Crypto Assistant")
st.markdown("Ask questions about cryptocurrency price history (past 1 year).")

# Get OpenAI key from .env or sidebar input
default_api_key = os.getenv("OPENAI_API_KEY")
openai_key = st.sidebar.text_input(
    "🔐 Enter your OpenAI API Key", type="password", value=default_api_key
)

if not openai_key:
    st.warning("OpenAI API key not found. Please enter it in the sidebar.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Optional clear button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")


# Load and process CSVs
@st.cache_data
def load_all_crypto_data(folder="crypto_data"):
    data_dict = {}
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for file in csv_files:
        try:
            coin = os.path.basename(file).split("_")[0].lower()
            df = pd.read_csv(file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            data_dict[coin] = df
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return data_dict


# Prepare docs for RAG
def load_crypto_documents(data_dict):
    documents = []
    for coin, df in data_dict.items():
        for _, row in df.iterrows():
            text = f"{coin.capitalize()} price on {row['timestamp'].date()}: ${row['price']:.2f}"
            documents.append(Document(page_content=text))
    return documents


# Build index
@st.cache_resource
def build_vector_store(api_key, data_dict):
    docs = load_crypto_documents(data_dict)
    if not docs:
        raise ValueError("No documents found for vector store. Check your CSV files.")
    else:
        print(f"📄 Loaded {len(docs)} documents before splitting")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "No text chunks created. Check your data or splitter settings."
        )
    else:
        print(f"✂️ Split into {len(chunks)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)


# Load CSVs and vector index
with st.spinner("🔍 Indexing crypto data..."):
    crypto_data = load_all_crypto_data()
    vectorstore = build_vector_store(openai_key, crypto_data)

# LangChain setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**❓You:** {q}")
        st.markdown(f"**🤖 GPT:** {a}")
        st.markdown("---")

# Input form
with st.form(key="query_form"):
    user_query = st.text_input(
        "🔎 Ask a question about crypto prices:", key="user_query"
    )
    submit_btn = st.form_submit_button("Ask")

# After the form
if submit_btn and user_query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(user_query)
        st.session_state.chat_history.append((user_query, answer))
        st.markdown("### 📘 Answer")
        st.write(answer)

        # Match coin for chart
        matched_coin = next(
            (coin for coin in crypto_data if coin in user_query.lower()), None
        )
        if matched_coin:
            st.markdown(f"### 📈 Price Chart: {matched_coin.capitalize()}")
            df = crypto_data[matched_coin]
            st.line_chart(df.set_index("timestamp")["price"])
        else:
            st.info(
                "💡 Tip: Ask about a specific coin (e.g., Bitcoin, Ethereum) to see a chart."
            )

    # ✅ Reset the input field by clearing session state
    st.session_state["user_query"] = ""
