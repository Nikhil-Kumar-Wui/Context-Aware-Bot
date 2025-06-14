# Context-Aware Chatbot with Local Semantic Search

A gemini based chatbot built using **TF-IDF + Cosine Similarity** to answer queries from uploaded `.txt` or `.pdf` files — ideal for answering university or document-specific questions without relying on cloud storage or external databases.

---

## What It Does

This chatbot can:
- Read and process text or PDF files uploaded by the user
- Accept natural language queries
- Retrieve the most relevant sentence(s) from the document using **TF-IDF vectorization**
- Match based on **cosine similarity** between query and document vectors
- Maintain basic context for multi-turn conversations (semi-stateless)

---

## Tools Used

| Tool/Library     | Purpose                                      |
|------------------|----------------------------------------------|
| Python           | Core programming language                    |
| Streamlit        | Web app framework for UI                     |
| scikit-learn     | TF-IDF Vectorizer and cosine similarity      |
| PyPDF2           | PDF text extraction                          |
| python-dotenv    | Environment variable management (local dev)  |
| Gemini API  | Can be used for LLM integration in future |

---

## How It Works

1. **Upload File** (.txt or .pdf)
2. **Preprocessing:** Clean and chunk the text
3. **Vectorization:** Convert document + query into TF-IDF vectors
4. **Similarity Search:** Use cosine similarity to rank and return the most relevant text chunk
5. **Semi-Context Awareness:** Previous queries can be optionally combined to improve relevance

---

## Local Setup


```bash
git clone https://github.com/your-username/context-aware-bot.git
cd context-aware-bot

pip install -r requirements.txt
GEMINI_API_KEY=your-api-key-here
streamlit run app.py
