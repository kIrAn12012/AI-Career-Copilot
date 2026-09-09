import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from dotenv import load_dotenv
import os
import numpy as np


# ---------- SETUP ----------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing from the .env file.")
    st.stop()

client = genai.Client(
    api_key=api_key
)


# ---------- COSINE SIMILARITY ----------

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ---------- APP TITLE ----------

st.title("📚 AI Study Assistant")
st.write("Upload a PDF and ask questions about it using AI.")


# ---------- CHAT HISTORY ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- CLEAR CHAT ----------

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


# ---------- NAME ----------

name = st.text_input("Enter your name:")

if name:
    st.write(f"Welcome {name}! 🚀")


# ---------- PDF UPLOAD ----------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    st.write("PDF uploaded successfully!")
    st.write(uploaded_file.name)


    # ---------- PDF → TEXT ----------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    if not text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

        st.stop()


    # ---------- TEXT → CHUNKS ----------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)


    # ---------- CHUNKS → EMBEDDINGS ----------

    embeddings = []

    for chunk in chunks:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )

        embeddings.append(
            result.embeddings[0].values
        )


    # ---------- STORE CHUNK + EMBEDDING ----------

    documents = []

    for i in range(len(chunks)):

        documents.append({
            "text": chunks[i],
            "embedding": embeddings[i]
        })


    # ---------- PDF INFORMATION ----------

    st.write("Number of chunks:", len(chunks))
    st.write("Number of embeddings:", len(embeddings))
    st.write("Documents stored:", len(documents))

    st.write(
        "First embedding size:",
        len(documents[0]["embedding"])
    )


    # ---------- USER QUESTION ----------

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    # ---------- ASK BUTTON ----------

    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # ---------- SAVE USER QUESTION ----------

            st.session_state.chat_history.append({
                "role": "user",
                "message": question
            })


            # ---------- QUESTION → EMBEDDING ----------

            question_result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=question
            )

            question_embedding = (
                question_result.embeddings[0].values
            )


            # ---------- SIMILARITY ----------

            similarities = []

            for document in documents:

                score = cosine_similarity(
                    question_embedding,
                    document["embedding"]
                )

                similarities.append(score)


            # ---------- TOP 3 RETRIEVAL ----------

            top_k = min(3, len(documents))

            top_indices = np.argsort(
                similarities
            )[-top_k:][::-1]


            relevant_chunks = []

            for index in top_indices:

                relevant_chunks.append({
                    "text": documents[index]["text"],
                    "score": similarities[index]
                })


            # ---------- DISPLAY TOP 3 ----------

            st.write("Top relevant chunks:")

            for i, chunk in enumerate(
                relevant_chunks
            ):

                st.write(
                    f"Chunk {i + 1}:"
                )

                st.write(
                    "Similarity:",
                    chunk["score"]
                )

                st.write(
                    chunk["text"]
                )


            # ---------- COMBINE TOP 3 CHUNKS ----------

            context = "\n\n".join(
                chunk["text"]
                for chunk in relevant_chunks
            )


            # ---------- PROMPT ----------

            prompt = f"""
You are an AI assistant that answers questions
using information retrieved from a PDF.

Use only the information provided in the PDF context.

PDF context:
{context}

User question:
{question}

Instructions:
- Answer clearly and simply.
- Use the PDF context to answer.
- Do not invent information.
- If the answer is not present in the PDF context,
  say that you could not find the answer in the PDF.
"""


            # ---------- GEMINI GENERATION ----------

            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                answer = response.text


                # ---------- SAVE AI ANSWER ----------

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": answer
                })


                # ---------- SHOW AI ANSWER ----------

                st.write("AI Answer:")
                st.write(answer)


            except Exception as e:

                st.error(
                    "Gemini is temporarily unavailable. "
                    "Please try again in a few minutes."
                )

                st.write(
                    "Technical error:",
                    str(e)
                )


            # ---------- SHOW CONTEXT ----------

            with st.expander(
                "View retrieved context"
            ):

                st.write(context)


# ---------- DISPLAY CHAT HISTORY ----------

if st.session_state.chat_history:

    st.subheader("Chat History")

    for chat in st.session_state.chat_history:

        with st.chat_message(chat["role"]):

            st.write(chat["message"])