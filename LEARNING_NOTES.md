## 12 July

### Learned
- Created virtual environment
- Activated venv
- Installed libraries
- Created requirements.txt
- requirements.txt
- .gitignore
- Streamlit basics
- st.title()
- st.write()
- st.text_input()
- st.file_uploader()

### Commands

python -m venv venv
→ Create virtual environment

.\venv\Scripts\Activate.ps1
→ Activate environment

pip install streamlit google-generativeai pypdf python-dotenv
→ Install required libraries

pip freeze > requirements.txt
→ Save installed libraries and versions

- Browser app can be made using Python.
- Streamlit reruns app after refresh.

========================================================
AI CAREER COPILOT — NOTES AFTER 12 JULY
31 JULY → 3 AUGUST
========================================================


📅 31 JULY — PROJECT RESUMPTION
--------------------------------------------------------

Project flow we are building:

PDF
 ↓
Extract text
 ↓
Split text into chunks
 ↓
Create embeddings
 ↓
Store embeddings
 ↓
User asks question
 ↓
Find relevant chunks
 ↓
Send relevant chunks + question to Gemini
 ↓
Gemini generates answer


The goal is to build a PDF-based AI Study Assistant
as the first part of AI Career Copilot.



📅 3 AUGUST — TEXT CHUNKING
--------------------------------------------------------

WHY DO WE NEED CHUNKS?

A large PDF can contain a lot of text.

Sending the entire PDF to AI every time is not ideal because:

1. It uses more tokens.
2. It can increase cost.
3. There can be lots of irrelevant information.
4. AI can focus better when given relevant information.

Therefore:

Full PDF text
      ↓
Smaller chunks


INSTALL TEXT SPLITTER

Terminal:

pip install langchain-text-splitters


IMPORT:

from langchain_text_splitters import RecursiveCharacterTextSplitter


CREATE SPLITTER:

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


chunk_size=500
→ approximately 500 characters per chunk.

chunk_overlap=50
→ neighbouring chunks have around 50 characters in common.

Overlap helps preserve context between chunks.


SPLIT THE TEXT:

chunks = splitter.split_text(text)

Now:

text
→ one large string

chunks
→ list of smaller text pieces.


Example:

chunks[0] → first chunk
chunks[1] → second chunk
chunks[2] → third chunk


len(chunks)
→ number of chunks.

chunks[0]
→ first chunk.


IMPORTANT IDEA:

We are NOT changing the PDF.

We are only dividing the extracted text into smaller pieces.



📅 3 AUGUST — EMBEDDINGS
--------------------------------------------------------

WHAT IS AN EMBEDDING?

An embedding converts text into a numerical representation.

Conceptually:

Text
 ↓
Embedding model
 ↓
[0.12, -0.43, 0.78, ...]


WHY?

We eventually need to find which PDF chunks are relevant
to the user's question.

Example:

Question:
"What is gradient descent?"

        ↓
Question embedding

        ↓
Compare with chunk embeddings

        ↓
Find relevant chunk(s)


IMPORTANT:

Embedding ≠ AI answer.

Embedding is a numerical representation used mainly
to compare/search for relevant information.


Example:

Chunk 1 → embedding 1
Chunk 2 → embedding 2
Chunk 3 → embedding 3

Later:

Question → question embedding

Then compare question embedding with chunk embeddings.



📅 3 AUGUST — GOOGLE GEMINI API
--------------------------------------------------------

INSTALL GOOGLE GENAI:

pip install google-genai


IMPORT:

from google import genai


API KEY:

We created a Google AI Studio API key.

The key is stored in:

.env


Example:

GEMINI_API_KEY=your_api_key


DO NOT put the API key directly inside app.py.


.gitignore already contains:

.env

Therefore Git will not upload the API key.


LOAD .ENV:

Install:

pip install python-dotenv


Imports:

from dotenv import load_dotenv
import os


Then:

load_dotenv()


Read API key:

os.getenv("GEMINI_API_KEY")


We tested this safely using:

bool(os.getenv("GEMINI_API_KEY"))

If it shows:

True

→ Python successfully found the API key.

Never display the actual API key in Streamlit.



GEMINI CLIENT:

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


client = our connection/interface to Google's AI services.

Conceptually:

Python app
     ↓
   client
     ↓
Google AI



CREATE AN EMBEDDING:

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks[0]
)


This takes the first PDF chunk:

chunks[0]

and sends it to the embedding model.

Conceptually:

chunks[0]
    ↓
embedding model
    ↓
numerical vector


Embedding values:

result.embeddings[0].values


Number of values:

len(result.embeddings[0].values)



========================================================
CURRENT UNDERSTANDING
========================================================

PDF
 ↓
Extract text                         ✅ DONE
 ↓
Split into chunks                    ✅ DONE
 ↓
Create embedding for a chunk         ✅ STARTED
 ↓
Create embeddings for ALL chunks     ← NEXT
 ↓
Store embeddings                     ← AFTER THAT
 ↓
User asks question
 ↓
Find relevant chunks
 ↓
Send relevant chunks + question
to Gemini
 ↓
Generate answer


========================================================
THINGS TO REMEMBER
========================================================

PdfReader
→ reads the PDF

RecursiveCharacterTextSplitter
→ divides large text into chunks

Embedding
→ converts text into numbers representing its meaning

genai.Client()
→ connection to Google's AI service

embed_content()
→ creates an embedding for text

RAG
→ retrieve relevant information and use it to generate an answer


I DON'T NEED TO MEMORIZE EVERY FUNCTION.

I need to remember the LOGIC:

PDF
→ text
→ chunks
→ embeddings
→ retrieve relevant chunks
→ Gemini
→ answer
========================================================