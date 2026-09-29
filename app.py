import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from dotenv import load_dotenv
import os
import numpy as np
import fitz
import pytesseract
from PIL import Image
import io
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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

st.title("🤖 AI Career Copilot")

st.write(
    "Your AI assistant for resumes, jobs, interview preparation, "
    "and study support."
)

@st.cache_data
def create_embeddings(chunks):

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


# ==========================================================
#                  AI STUDY ASSISTANT
# ==========================================================

# ---------- CHAT HISTORY SETUP ----------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ---------- HEADER ----------

st.header("📚 AI Study Assistant")
st.write("Upload study material and ask questions about it.")


# ---------- NAME ----------

name = st.text_input("Enter your name:")

if name:

    st.write(f"Welcome {name}! 🚀")


# ---------- PDF UPLOAD ----------

uploaded_file = st.file_uploader(
    "Upload Study Material PDF",
    type=["pdf"],
    key="study_pdf_uploader"
)


if uploaded_file:

    st.write("PDF uploaded successfully!")
    st.write(uploaded_file.name)

# ---------- PDF → TEXT ----------

    reader = PdfReader(uploaded_file)
    text = ""

# First try normal PDF text extraction
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

# If no text was found, use OCR
    if not text.strip():
        st.info("No selectable text found. Running OCR...")

        uploaded_file.seek(0)

        pdf = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        ocr_text = []

        for page in pdf:
            pix = page.get_pixmap(dpi=200)

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            page_text = pytesseract.image_to_string(image)

            if page_text:
                ocr_text.append(page_text)

        text = "\n".join(ocr_text)

# Final check
    if not text.strip():
        st.error("Could not extract text from this PDF.")
    else:
        st.success("PDF text extracted successfully!")
    


        # ---------- TEXT → CHUNKS ----------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)
        st.write("OCR text length:", len(text))
        st.write("Number of chunks:", len(chunks))


        # ---------- CHUNKS → EMBEDDINGS ----------

        embeddings = create_embeddings(chunks)
        
        
        # ---------- STORE CHUNK + EMBEDDING ----------

        documents = []

        for i in range(len(chunks)):

            documents.append({
                "text": chunks[i],
                "embedding": embeddings[i]
            })


        # ---------- PDF INFORMATION ----------

        st.write(
            "Number of chunks:",
            len(chunks)
        )

        st.write(
            "Number of embeddings:",
            len(embeddings)
        )

        st.write(
            "Documents stored:",
            len(documents)
        )

        if documents:

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

                top_k = min(
                    3,
                    len(documents)
                )

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

                st.write(
                    "Top relevant chunks:"
                )

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


# ==========================================================
#                    CHAT HISTORY
# ==========================================================

if st.session_state.chat_history:

    st.subheader("💬 Chat History")


    # ---------- CLEAR CHAT ----------

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


    # ---------- DISPLAY CHAT HISTORY ----------

    for chat in st.session_state.chat_history:

        with st.chat_message(chat["role"]):

            st.write(chat["message"])


# ==========================================================
#                    RESUME ANALYZER
# ==========================================================

st.header("💼 Resume Analyzer")
st.write(
    "Upload your resume to analyze your skills, strengths, weaknesses, "
    "and areas for improvement."
)


# ---------- RESUME UPLOAD ----------

resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
    key="resume_uploader"
)


# ---------- RESUME TEXT EXTRACTION ----------

resume_text = ""

if resume_file:

    st.success("Resume uploaded successfully!")
    st.write(resume_file.name)

    reader = PdfReader(resume_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            resume_text += page_text + "\n"

    if not resume_text.strip():

        st.error("Could not extract text from this resume.")
        st.stop()

    st.write("Resume text extracted successfully!")


# ---------- RESUME ANALYSIS PROMPT ----------

if resume_text.strip():

    prompt = f"""
You are an AI career assistant.

Analyze the following resume and provide useful, honest feedback.

Resume:
{resume_text}

Give an overall resume score from 0 to 100 based on:
- skills
- projects
- experience
- clarity
- relevance for internships/jobs

Start your response with:

Resume Score: <score>/100

Explain briefly why you gave this score, considering:
- Technical skills
- Projects
- Experience
- Resume clarity
- Relevance for internships/jobs

Then give your analysis in the following sections:

1. Skills
- List the technical and soft skills found in the resume.

2. Strengths
- Identify the strongest parts of the resume.

3. Weaknesses
- Identify areas that could be improved.

4. Recommended Skills
- Suggest important skills the candidate could learn based on their current profile.

5. Resume Improvements
- Give specific suggestions to make the resume stronger for internships and jobs.

Keep the analysis clear, practical, and based only on the information present in the resume.
"""


    # ---------- ANALYZE RESUME ----------

    if st.button("Analyze Resume"):

        with st.spinner("Analyzing your resume..."):

            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                analysis = response.text

                st.subheader("📊 Resume Analysis")
                st.markdown(analysis)

            except Exception as e:

                st.error(
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                )

                st.write(
                    "Technical error:",
                    str(e)
                )


# ==========================================================
#                 JOB DESCRIPTION MATCHER
# ==========================================================

st.header("🎯 Job Description Matcher")
st.write(
    "Add the job description you want to target. "
    "You can paste it, upload a PDF, or provide a job posting URL."
)


# ---------- PASTE JOB DESCRIPTION ----------
st.subheader("📝 Option 1: Paste Job Description")
job_description = st.text_area(
    "Paste Job Description",
    height=250
)
# job description link
st.subheader("🔗 Option 2: Job Posting URL")
jd_link = st.text_input(
    "🔗 Or paste Job Description Link",
    placeholder="https://example.com/job"
)
# ---------- JOB DESCRIPTION LINK → TEXT ----------

jd_link_text = ""

if jd_link.strip():

    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            jd_link,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        st.write("HTML received:", len(response.text))

        for element in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            element.decompose()

        jd_link_text = soup.get_text(
            separator=" ",
            strip=True
        )

        if not jd_link_text:
            st.error("Could not extract text from this URL.")
        else:
            st.success("Job Description text extracted from URL!")

    except Exception as e:

        st.error("Could not extract Job Description from this URL.")

        st.write("Technical error:", str(e))


# ---------- JOB DESCRIPTION PDF ----------
st.subheader("📄 Option 3: Upload Job Description PDF")
job_pdf = st.file_uploader(
    "Or upload Job Description PDF",
    type=["pdf"],
    key="job_pdf_uploader"
)


# ---------- JD PDF → TEXT ----------

jd_pdf_text = ""

if job_pdf:

    st.success("Job Description PDF uploaded successfully!")
    st.write(job_pdf.name)

    reader = PdfReader(job_pdf)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            jd_pdf_text += page_text + "\n"

    if not jd_pdf_text.strip():

        st.error("Could not extract text from the JD PDF.")
        st.stop()

    st.write("Job Description text extracted successfully!")

    

    # ---------- COMBINE JD INPUT ----------

if job_description.strip():

    jd_text = job_description

elif job_pdf:

    jd_text = jd_pdf_text

elif jd_link.strip():

    jd_text = jd_link_text

else:

    jd_text = ""


# ---------- RESUME + JD EMBEDDINGS ----------

if resume_text.strip() and jd_text.strip():

    resume_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=resume_text
    )

    resume_embedding = (
        resume_result.embeddings[0].values
    )


    jd_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=jd_text
    )

    jd_embedding = (
        jd_result.embeddings[0].values
    )

    st.write(
        "Resume and JD embeddings created successfully!"
    )
    match_score = cosine_similarity(
        resume_embedding,
        jd_embedding
    )

    st.subheader("🎯 Resume–Job Match")

    match_percentage = round(match_score * 100)

    st.metric(
        "Semantic Match",
        f"{match_percentage}%"
    )

    st.caption(
        "This score measures semantic similarity between your resume "
        "and the job description. It is not an ATS score, hiring probability, "
        "or guarantee of selection."
    )
    # ---------- GEMINI JOB MATCH ANALYSIS ----------

    match_prompt = f"""
    You are an AI career assistant.

    Compare the following resume with the job description.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {jd_text}

    Analyze how well the resume matches the job.

    Give the analysis in exactly these sections:

    1. Matching Skills
    - List the important skills from the job description that are already present in the resume.

    2. Missing Skills
    - List important skills required by the job description that are missing from the resume.

    3. Resume Strengths for This Job
    - Explain the strongest parts of the resume for this particular role.

    4. Recommendations
    - Suggest what the candidate should learn, improve, or add to become a stronger match.

    5. Overall Assessment
    - Give a short honest assessment of how suitable the resume is for this job.

    Do not invent skills, experience, or projects that are not present in the resume.
    Base the analysis only on the provided resume and job description.
    """

    if st.button("Analyze Job Match"):

        with st.spinner("Analyzing resume against job description..."):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=match_prompt
                )

                match_analysis = response.text

                st.subheader("📊 Job Match Analysis")

                st.markdown(match_analysis)

            except Exception as e:

                st.error(
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                )

                st.write("Technical error:", str(e))

                match_percentage = round(match_score * 100)

                st.subheader("🎯 Resume-JD Match Score")

                st.metric(
                    "Semantic Match",
                    f"{match_percentage}/100"
                )

                if match_percentage >= 80:
                     st.success("Strong match — your resume aligns well with this job.")

                elif match_percentage >= 60:
                    st.warning("Moderate match — some important skills may be missing.")

                else:
                    st.error("Low match — the resume has limited alignment with this job.")

# ---------- INTERVIEW PREP ----------

st.header("🎤 Interview Prep")

st.write(
    "Generate personalized interview questions using your uploaded resume "
    "and the current job description."
)

# Check whether the existing Resume and JD are available
resume_available = bool(resume_text.strip())
jd_available = bool(jd_text.strip())

col1, col2 = st.columns(2)

with col1:
    st.write(
        f"**Resume:** {'✅ Available' if resume_available else '❌ Not available'}"
    )

with col2:
    st.write(
        f"**Job Description:** {'✅ Available' if jd_available else '❌ Not available'}"
    )

generate_questions = st.button(
    "🎯 Generate Interview Questions",
    disabled=not (resume_available and jd_available)
)

if generate_questions:


    interview_prompt = f"""
You are an AI interview coach.

Your task is to create a personalized interview preparation set
for the candidate based ONLY on the candidate's resume and the
provided job description.

====================
CANDIDATE RESUME
====================

{resume_text}

====================
JOB DESCRIPTION
====================

{jd_text}

====================
TASK
====================

First understand:
1. The role and responsibilities from the job description.
2. The important technical skills and requirements.
3. The candidate's skills, projects, education, and experience from the resume.
4. The overlap between the resume and the job requirements.
5. Important job requirements that are not clearly demonstrated in the resume.

Then generate exactly 5 personalized interview questions.

Question distribution:

1. Technical Question
   - Based on an important technical requirement from the job description.
   - Prefer something relevant to the candidate's demonstrated skills.

2. Technical / Gap Question
   - Based on an important job requirement that is weakly demonstrated
     or not clearly demonstrated in the resume.
   - Do not assume the candidate has this skill.

3. Resume / Project Question
   - Ask about a specific project, skill, education item, or experience
     mentioned in the resume.
   - The question should test the candidate's actual understanding.

4. Behavioral Question
   - Relevant to the responsibilities of the job.
   - Keep it suitable for a student/intern candidate.

5. HR / Motivation Question
   - Relevant to the role and the candidate's background.
   - Example areas include interest in the role, career goals,
     learning ability, or motivation.

For EACH question provide:

Question:
<question>

Why this is asked:
<what the interviewer is trying to evaluate>

What to focus on:
<short guidance on what the candidate should cover in their answer>

Important rules:

- Use BOTH the resume and job description.
- Do not invent any project, skill, experience, achievement, or qualification.
- Never claim that the candidate has a skill unless it appears in the resume.
- If a requirement is missing from the resume, treat it as a gap rather than
  pretending the candidate knows it.
- Make questions specific to this candidate and this job.
- Avoid generic questions that could apply to any candidate.
- Keep the questions realistic for an internship/student interview.
- Keep explanations concise and useful.
"""

    with st.spinner("Analyzing your resume and job description..."):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=interview_prompt
            )

            st.subheader("📝 Personalized Interview Questions")

            st.markdown(response.text)

        except Exception as e:

            st.error(
                "Gemini is temporarily unavailable. Please try again."
            )

            st.write(
                "Technical error:",
                str(e)
            )

# ---------- RESUME IMPROVEMENT ----------

st.header("✨ Resume Improvement")

st.write(
    "Get personalized resume improvement suggestions based on "
    "your current resume and the selected job description."
)

resume_available = bool(resume_text.strip())
jd_available = bool(jd_text.strip())

col1, col2 = st.columns(2)

with col1:
    st.write(
        f"**Resume:** {'✅ Available' if resume_available else '❌ Not available'}"
    )

with col2:
    st.write(
        f"**Job Description:** {'✅ Available' if jd_available else '❌ Not available'}"
    )

improve_resume = st.button(
    "✨ Improve My Resume",
    disabled=not (resume_available and jd_available)
)

if improve_resume:

    resume_improvement_prompt = f"""
You are an expert resume improvement assistant.

Analyze the candidate's resume against the provided job description.

====================
CANDIDATE RESUME
====================

{resume_text}

====================
JOB DESCRIPTION
====================

{jd_text}

====================
TASK
====================

Identify how the candidate can improve their resume specifically
for this job.

Provide exactly these 4 sections:

### 1. Missing or Weak Skills
List important skills or requirements from the job description that
are missing or weakly demonstrated in the resume.

For each one:
- Skill/requirement
- Status: Missing / Weakly demonstrated
- Recommendation

Do NOT assume the candidate has a skill that is not present in the resume.

### 2. Resume Bullet Improvements
Identify up to 3 resume bullets or statements that could be improved.

For each:
- Current statement
- Suggested improved version
- Why it is better

IMPORTANT:
Only rewrite using information already present in the resume.
Do not invent numbers, achievements, technologies, or experiences.

### 3. Job-Specific Recommendations
Give 3-5 specific recommendations for improving the resume
for this particular job.

Focus on:
- Relevant skills
- Projects
- Technical terminology
- Resume structure
- Relevance to the job requirements

### 4. Important Things NOT to Add
Mention skills, experiences, or achievements from the job description
that the candidate should NOT falsely add to their resume unless
they genuinely have them.

====================
RULES
====================

- Use BOTH the resume and job description.
- Do not invent information.
- Do not exaggerate the candidate's experience.
- Do not recommend adding a skill merely because it appears in the JD.
- Keep recommendations practical for a student/intern.
- Prioritize the most relevant improvements.
"""

    with st.spinner("Analyzing your resume for this job..."):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=resume_improvement_prompt
            )

            st.subheader("✨ Resume Improvement Suggestions")

            st.markdown(response.text)

        except Exception as e:

            st.error(
                "Gemini is temporarily unavailable. Please try again."
            )

            st.write(
                "Technical error:",
                str(e)
            )