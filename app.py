import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import fitz
import google.generativeai as genai


# Google Gemini API key

api_key=st.secrets["GEMINI"]["API_KEY"]
genai.configure(api_key=api_key)

model=genai.GenerativeModel("gemini-2.0-flash")


# Streamlit configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon=":mag_right:", layout="centered")

st.title("Retrieval-Based Chatbot with Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

# if "history" not in st.session_state:
#     st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# helper function to get relevant chunks
def get_relevant_chunks(query):
    vec=st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunks_vectors).flatten()
    top_indices=similarities.argsort()[-3:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])

def build_prompt(query):
    chat_history="\n".join(
        [f"User: {u}\nBot: {b} " for u, b in st.session_state.history[: -3]]
    )
    context=get_relevant_chunks(query)
    prompt=f"""
    You are a helpful assistant. That answers questions based on the provided context.

context: {context}
Chat history: {chat_history}
User: {query}
Bot:
"""
    return prompt

st.sidebar.title("Upload PDF")
uploaded_file=st.sidebar.file_uploader("Upload a Text or PDF file", type=["txt", "pdf"])

def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
    

    chunks = chunk_text(text)
    vectorizer=TfidfVectorizer().fit(chunks)
    chunks_vectors=vectorizer.transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks_vectors = chunks_vectors

if uploaded_file:
    file_type=uploaded_file.type.split("/")[-1]
    process_file(uploaded_file, file_type)
    st.sidebar.success("File processed successfully!")

# Chat Interface

st.subheader("Ask anything:")
user_query=st.text_input("Your Question:")

if st.button("Ask") and user_query and "vectorizer" in st.session_state:
    prompt=build_prompt(user_query)
    try:
        response=model.generate_content(prompt)
        bot_reply=response.text.strip()
        st.session_state.history.append((user_query, bot_reply))
        st.success("Response generated successfully!")
    except Exception as e:
        st.error(f"Error generating response: {e}")

if st.session_state.history:
    st.subheader("Conversation")
    for user,bot in reversed(st.session_state.history):
        st.markdown(f"**Me:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")


