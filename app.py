import os
import glob
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Load env variables
load_dotenv()
st.set_page_config(page_title="🪙 Crypto AI Assistant", layout="centered")
st.markdown("# 🪙 CryptoGPT — Your Smart Crypto Assistant")
st.markdown("Ask questions about cryptocurrency price history (past 1 year).")

# Get OpenAI key from .env or sidebar input
default_api_key = os.getenv("OPENAI_API_KEY")
openai_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password", value=default_api_key)

if not openai_key:
    st.warning("OpenAI API key not found. Please enter it in the sidebar.")
    st.stop()

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
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)

# Load CSVs and vector index
with st.spinner("🔍 Indexing crypto data..."):
    crypto_data = load_all_crypto_data()
    vectorstore = build_vector_store(openai_key, crypto_data)

# LangChain setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Input form
with st.form(key="query_form"):
    user_query = st.text_input("🔎 Ask a question about crypto prices:", key="user_query")
    submit_btn = st.form_submit_button("Ask")

# After the form
if submit_btn and user_query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(user_query)
        st.markdown("### 📘 Answer")
        st.write(answer)

        # Match coin for chart
        matched_coin = next((coin for coin in crypto_data if coin in user_query.lower()), None)
        if matched_coin:
            st.markdown(f"### 📈 Price Chart: {matched_coin.capitalize()}")
            df = crypto_data[matched_coin]
            st.line_chart(df.set_index("timestamp")["price"])
        else:
            st.info("💡 Tip: Ask about a specific coin (e.g., Bitcoin, Ethereum) to see a chart.")

    # ✅ Reset the input field by clearing session state
    st.session_state["user_query"] = ""
