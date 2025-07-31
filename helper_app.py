import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from dotenv import load_dotenv
from langdetect import detect
import easyocr
import numpy as np
import fitz
from io import BytesIO
import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    'gemini-1.5-flash-8b-001',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
)

def detect_language(text):
    return detect(text)

def create_pdf(content):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    for line in content.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line.strip())
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer


def read_image(file, lang):
    image = Image.open(file)
    image_np = np.array(image)
    reader = easyocr.Reader([lang, 'en'], gpu=False)
    result = reader.readtext(image_np, detail=0)
    return ' '.join(result)

def extract_text_from_file(file):
    try:
        if file.name.endswith(".pdf"):
            text = ""
            pdf_bytes = file.read()
            if not pdf_bytes:
                st.error("Error: Uploaded PDF is empty.")
                return None
            try:
                reader = PdfReader(BytesIO(pdf_bytes))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n\n"
            except:
                pass
            if not text:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in doc:
                        t = page.get_text("text")
                        if t:
                            text += t + "\n\n"
                except:
                    pass
            if not text:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in doc:
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        lang = detect_language(read_image(BytesIO(img_bytes), 'en'))
                        text += read_image(BytesIO(img_bytes), lang) + "\n\n"
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    return None
            return text

        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")

        elif file.name.endswith(".docx"):
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif file.name.endswith((".jpg", ".jpeg", ".png")):
            temp_text = read_image(file, 'en')
            lang = detect_language(temp_text)
            return read_image(file, lang)

        else:
            st.error("Unsupported file type.")
            return None

    except Exception as e:
        st.error(f"Text extraction failed: {e}")
        return None

def summarize_with_gemini(text, prompt_template):
    try:
        prompt = prompt_template.format(text=text)
        response = model.generate_content(prompt)
        return response.text if response.text else None
    except Exception as e:
        st.error(f" API Error: {e}")
        return None


def main():
    st.set_page_config(page_title="Job Recommender", layout="wide")
    st.title("📄 Smart Job Recommendation Tool")

    st.markdown("""
    Upload a single resume (**PDF, DOCX, TXT, XML, Image**) and enter the job description.
    Our system will analyze the resume and recommend suitability for the role.
    """)

    st.markdown("### 🧾 Job Description or Role Requirement")
    st.markdown("_Please describe the role requirements or paste the job description below._")

    job_input = st.text_area(
        label="Job Description Input (hidden label)",  # Required for accessibility
        placeholder="e.g. Data Scientist with 5+ years of ML, DL, NLP, LLM experience...",
        height=150,
        label_visibility="collapsed"  # Hides the label visually but keeps it for screen readers
    )


    default_prompt = (
        """
        You are an expert HR recruiter and resume screener.\n\n
        Your job is to evaluate whether the following resume is a good match for the given job description:\n{requirement}\n\n
        If the job description is unclear, meaningless, too short, or irrelevant, respond with:\n
        "```\n"
        "Decision: NO\n"
        "Score: 0/100\n"
        "Summary: The job description is invalid or unclear.\n"
        "Matched Skills:\n- None\n"
        "Missing or Weak Areas:\n- All (Invalid job description)\n"
        "```\n\n"
        "Otherwise, "Please analyze the Candidate Resume:\n{text}\n\n and provide:\n"
        "1. A clear decision: YES or NO\n"
        "2. A match score between 0 and 100\n"
        "3. A brief summary (2–3 lines)\n"
        "4. Matched Skills\n"
        "5. Missing Skills\n\n"
        "**Format your response as below:**\n"
        "```\n"
        "Decision: YES or NO\n"
        "Score: XX/100\n"
        "Summary: <2–3 line explanation>\n"
        "Matched Skills:\n- ...\n- ...\n"
        "Missing or Weak Areas:\n- ...\n- ...\n"
        "```
        """
    )




    file = st.file_uploader(
        "📤 Upload a Resume",
        type=["pdf", "txt", "docx", "xml", "jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if file and st.button("🔍 Suggest Suitability"):
        if not job_input.strip():
            st.warning("⚠️ Please enter the job description.")
        elif len(job_input.split()) < 5:
            st.warning("⚠️ Please provide a more detailed job description.")
        elif not file:
            st.warning("⚠️ Please upload a resume file.")
        else:
            requirement = job_input.strip()
            text = extract_text_from_file(file)
            if text:
                final_prompt = default_prompt.format(requirement=f"{requirement}", text=f"{text}")
                with st.spinner("🔍 Analyzing resume..."):
                    summary = summarize_with_gemini(text, final_prompt)

                st.subheader(f"📝 Recommendation for: `{file.name}`")

                if summary:
                    # ✅ Extract score and decision
                    score_match = re.search(r"Score:\s*(\d+)", summary)
                    decision_match = re.search(r"Decision:\s*(YES|NO)", summary, re.IGNORECASE)

                    score = int(score_match.group(1)) if score_match else 0
                    decision = decision_match.group(1).upper() if decision_match else "NO"

                    # ✅ Decision banner
                    if decision == "YES":
                        st.success(f"✅ Decision: {decision} — Suitable Candidate")
                    else:
                        st.error(f"❌ Decision: {decision} — Not a Suitable Match")

                    # ✅ Progress bar
                    st.markdown("### 🎯 Match Score")
                    st.progress(score)
                    st.metric(label="Score", value=f"{score}/100")

                    # ✅ Highlight sections: Summary, Matched & Missing Skills
                    summary_lines = summary.splitlines()
                    matched_skills = []
                    missing_skills = []
                    show_summary = []

                    collecting = None
                    for line in summary_lines:
                        if line.startswith("Summary:"):
                            show_summary.append(line.replace("Summary:", "").strip())
                        elif line.startswith("Matched Skills:"):
                            collecting = "matched"
                        elif line.startswith("Missing") or line.startswith("Weak Areas"):
                            collecting = "missing"
                        elif collecting == "matched" and line.strip().startswith("-"):
                            matched_skills.append(line.strip("- ").strip())
                        elif collecting == "missing" and line.strip().startswith("-"):
                            missing_skills.append(line.strip("- ").strip())

                    st.markdown("### ✍️ Summary")
                    st.info(" ".join(show_summary))

                    st.markdown("### 📊 Skills Match Analysis")

                    if matched_skills or missing_skills:
                        # Prepare DataFrame
                        skill_data = pd.DataFrame({
                            "Skill Type": ["Matched"] * len(matched_skills) + ["Missing"] * len(missing_skills),
                            "Skill": matched_skills + missing_skills
                        })

                        # Countplot: Horizontal bars, styled
                        fig, ax = plt.subplots(figsize=(6, 2.5))  # Adjust width and height here
                        sns.set_style("whitegrid")
                        sns.countplot(
                            data=skill_data,
                            y="Skill Type",
                            palette={"Matched": "#73C6B6", "Missing": "#F1948A"},
                            ax=ax
                        )

                        # Add count labels on the bars
                        for container in ax.containers:
                            ax.bar_label(container, label_type='edge', padding=3)

                        ax.set_title("Matched vs Missing Skills", fontsize=12, weight='bold')
                        ax.set_xlabel("Count")
                        ax.set_ylabel("")
                        st.pyplot(fig)

                    else:
                        st.warning("⚠️ No skill details found to visualize.")

                    # ✅ Optionally show raw model response
                    with st.expander("🔎 View Full Gemini Response"):
                        st.code(summary)

                    # ✅ PDF Download Button (separate expander or just below)
                    with st.expander("📥 Download Recommendation as PDF"):
                        pdf_buffer = create_pdf(summary)
                        st.download_button(
                            label="Download as PDF",
                            data=pdf_buffer,
                            file_name=f"{file.name}_recommendation.pdf",
                            mime="application/pdf"
                        )

                else:
                    st.error("❌ Could not generate a summary.")
            else:
                st.error("❌ Could not extract text from the uploaded file.")



if __name__ == "__main__":
    main()
