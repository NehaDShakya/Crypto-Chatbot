import glob
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from zep_cloud.client import Zep

# ---- Load .env and API keys ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZEP_API_KEY = os.getenv("ZEP_API_KEY")

# ---- Zep client ----
client = Zep(api_key=ZEP_API_KEY)
SESSION_ID = "streamlit-session"

# ---- Streamlit config ----
st.set_page_config(page_title="🪙 Crypto AI Assistant", layout="centered")
st.markdown("# 🪙 CryptoGPT — Your Smart Crypto Assistant")
st.markdown("Ask questions about cryptocurrency price history (past 1 year).")


# ---- Load and prepare crypto CSV data ----
@st.cache_data
def load_all_crypto_data(folder="data"):
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


# ---- Load data ----
with st.spinner("🔍 Indexing crypto data..."):
    crypto_data = load_all_crypto_data()
    vectorstore = build_vector_store(OPENAI_API_KEY, crypto_data)
    retriever = vectorstore.as_retriever()

# ---- Initialize OpenAI chat model ----
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


# ---- Retrieve past messages from Zep ----
def get_chat_history():
    try:
        messages = client.memory.get(session_id=SESSION_ID)
        return [
            (
                f"User: {m['content']}"
                if m["role"] == "user"
                else f"Assistant: {m['content']}"
            )
            for m in messages
        ]
    except Exception:
        # Session probably doesn't exist yet
        return []


# ---- Build prompt with history ----
def build_prompt(history, question, context):
    chat_history_text = "\n".join(history[-5:])  # Limit to last 5
    return f"""
You are a cryptocurrency assistant. Use the following information to answer the question.

Context:
{context}

Recent Chat History:
{chat_history_text}

Current User Question:
{question}
"""


# ---- Streamlit input field ----
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False


user_query = st.text_input(
    "🔎 Ask a question about crypto prices:",
    value="" if st.session_state.clear_input else "",
    key="user_query",
)

if st.button("Ask") and user_query:
    with st.spinner("💬 Thinking..."):
        # Get relevant context from vectorstore
        docs = retriever.get_relevant_documents(user_query)
        CONTEXT = "\n".join(doc.page_content for doc in docs[:5])

        # Pull Zep history
        chat_history = get_chat_history()

        # Build prompt
        FULL_PROMPT = build_prompt(chat_history, user_query, CONTEXT)

        # Ask GPT
        response = llm([HumanMessage(content=FULL_PROMPT)])
        answer = response.content

        # Save to Zep memory
        try:
            client.memory.add(
                session_id=SESSION_ID,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": answer},
                ],
            )
        except Exception as e:
            st.error(f"Error saving to memory: {e}")

        # Show answer
        st.markdown("### 📘 Answer")
        st.write(answer)

        # Show chart if coin detected
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

    # Reset the input field
    st.session_state.clear_input = True
