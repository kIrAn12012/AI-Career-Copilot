import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Function to calculate similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


st.title("📚 AI Study Assistant")
st.write("Upload PDFs and chat with them using AI!")

# ---------- CHAT HISTORY ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

name = st.text_input("Enter your name:")

if name:
    st.write(f"Welcome {name}! 🚀")


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
        text += page.extract_text() + "\n"


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


    # ---------- TESTING ----------

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


    # ---------- QUESTION → EMBEDDING ----------

    if question:

        st.write("Your question:", question)

        st.session_state.chat_history.append({
            "role": "user",
            "message": question
        })

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

        top_k = 3

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

        for i, chunk in enumerate(relevant_chunks):

            st.write(f"Chunk {i + 1}:")
            st.write("Similarity:", chunk["score"])
            st.write(chunk["text"])


        # ---------- COMBINE TOP 3 CHUNKS ----------

        context = "\n\n".join(
            chunk["text"]
            for chunk in relevant_chunks
        )


        # ---------- GENERATE ANSWER ----------

        prompt = f"""
Answer the user's question using the information
from the PDF below.

PDF information:
{context}

User question:
{question}

If the answer is not present in the PDF information,
say that you could not find the answer in the PDF.
"""


        # response = client.models.generate_content(
        #     model="gemini-3.8-flash",
        #     contents=prompt
        # )


        # st.write("AI Answer:")
        # st.write(response.text)
        st.write("Context sent to AI:")
        st.write(context)



st.write("Chat History:")

for chat in st.session_state.chat_history:
    st.write(chat["role"], ":", chat["message"])