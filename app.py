import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
st.title("📚 AI Study Assistant")
st.write("upload PDFs and chat with them using AI!")
name = st.text_input("Enter your name : ")
if name:
    st.write(f"Welcome {name}! 🚀")
uploaded_file = st.file_uploader(
    "upload your pdf",
    type=["pdf"]
)
if uploaded_file:
    st.write("PDF uploaded successfully!")
    st.write(uploaded_file.name)

    reader =PdfReader(uploaded_file)
    #st.write(len(reader.pages))
    #st.write(reader.pages[0])
    #st.write(reader.pages[0].extract_text())
    text =""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    splitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)#only created splitter not splitted yet
    chunks =splitter.split_text(text)#actually splitting
    st.write("number of chunks : ",len(chunks))
    st.write("first chunk")
    st.write(chunks[0])

    st.write("PDF text extracted successfully!")
    st.write(text[:1000])

