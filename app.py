import streamlit as st
import os
import pandas as pd
import numpy as np
from utils.parse_resume import extract_text
from utils.vector_store import index_resume, query_jd, embed_text
from dotenv import load_dotenv
from typing import List
from html import escape

load_dotenv()

# ------------------------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Agent",
    layout="wide",
    page_icon="📄"
)

# ------------------------------------------------------------------------------------------------
# CUSTOM CSS (Modern UI)
# ------------------------------------------------------------------------------------------------
st.markdown("""
<style>

/* GLOBAL FONT */
html, body, p, div {
    font-family: 'Segoe UI', sans-serif;
}

/* Background */
body {
    background: linear-gradient(135deg, #E7F1FF 0%, #FFFFFF 100%);
}

/* Card Container */
.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    transition: all 0.2s ease-in-out;
    margin-bottom: 26px;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0px 10px 26px rgba(0,0,0,0.10);
}

/* Main Title */
.big-title {
    text-align:center;
    font-size:44px;
    font-weight:800;
    color:#2457A5;
    letter-spacing:-1px;
}

/* Subtitle */
.subtitle {
    text-align:center;
    font-size:18px;
    color:#444;
    margin-top:-10px;
}

/* Section Title */
.section-title {
    font-size:24px;
    font-weight:700;
    color:#15397F;
    margin-top: 20px;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 10px 25px;
    background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC);
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.18);
}

/* Textarea */
textarea {
    border-radius: 10px !important;
}

/* File uploader */
.css-1r6slb0 {
    padding: 12px !important;
    border-radius: 12px !important;
    background: white !important;
    border: 2px dashed #B9C9FF !important;
}

/* Download Button */
.stDownloadButton>button {
    border-radius: 10px;
    padding: 8px 20px;
    background: #345B9A;
    color: white;
    border: none;
}
.stDownloadButton>button:hover {
    background: #2E4C80;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------------------------
st.markdown("<h1 class='big-title'>📄 AI Resume Screening Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload resumes • Paste job description • Get ranked ATS scores</p>", unsafe_allow_html=True)
st.write("---")

# ------------------------------------------------------------------------------------------------
# SKILLS DB
# ------------------------------------------------------------------------------------------------
COMMON_SKILLS = [
    "python","java","c++","c#","javascript","react","node.js","node","flask","django","streamlit",
    "sql","mysql","postgresql","mongodb","aws","azure","gcp","docker","kubernetes","git","html",
    "css","tensorflow","pytorch","scikit-learn","nlp","computer vision","rest api","graphql",
    "excel","power bi","tableau","spring","hibernate","linux"
]
SKILLS_LOWER = [s.lower() for s in COMMON_SKILLS]

# ------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------
def extract_skills_from_text(text: str, skills_list: List[str]):
    text_lower = text.lower()
    return sorted({skill for skill in skills_list if skill in text_lower})

def compute_cosine_sim(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def highlight_keywords(text: str, keywords: List[str], max_len: int = 700):
    snippet = text[:max_len]
    safe_text = escape(snippet)
    for kw in sorted(set(keywords), key=lambda x: -len(x)):
        safe_text = safe_text.replace(escape(kw), f"<mark>{escape(kw)}</mark>")
    return safe_text

def compute_ats_score(skill_match_ratio, keyword_coverage, similarity_pct):
    return round(0.5 * (skill_match_ratio * 100) + 0.3 * (keyword_coverage * 100) + 0.2 * similarity_pct, 1)

# ------------------------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
top_k = st.sidebar.slider("Top K results", 1, 20, 5)
show_full_resume = st.sidebar.checkbox("Show full resume", False)

# ------------------------------------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------------------------------------
st.markdown("<h3 class='section-title'>1️⃣ Job Description</h3>", unsafe_allow_html=True)
job_description = st.text_area("Paste the Job Description:", height=180)

st.markdown("<h3 class='section-title'>2️⃣ Upload Resumes</h3>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

process_btn = st.button("🚀 Process Resumes", use_container_width=True)

# ------------------------------------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------------------------------------
if process_btn:
    if not job_description:
        st.error("❗ Please paste a job description.")
    elif not uploaded_files:
        st.error("❗ Please upload at least one resume.")
    else:

        with st.spinner("🔍 Screening candidates... please wait..."):

            # Save, parse, index resumes
            for f in uploaded_files:
                temp_path = f"tmp_{f.name}"
                with open(temp_path, "wb") as tmp:
                    tmp.write(f.getbuffer())

                text = extract_text(temp_path)
                index_resume(f.name, text)
                os.remove(temp_path)

            matches = query_jd(job_description, top_k=len(uploaded_files))
            ids = matches.get("ids", [[]])[0]
            docs = matches.get("documents", [[]])[0]
            jd_emb = embed_text(job_description)

            rows = []
            for i, doc_text in enumerate(docs):
                cid = ids[i]
                doc_emb = embed_text(doc_text)

                cos = compute_cosine_sim(jd_emb, doc_emb)
                similarity_pct = round(((cos + 1) / 2) * 100, 1)

                jd_skills = extract_skills_from_text(job_description, SKILLS_LOWER)
                resume_skills = extract_skills_from_text(doc_text, SKILLS_LOWER)

                matched = sorted(set(jd_skills).intersection(resume_skills))
                missing = sorted(set(jd_skills).difference(resume_skills))

                keyword_coverage = len(matched) / len(jd_skills) if jd_skills else 0
                skill_match_ratio = len(matched) / len(jd_skills) if jd_skills else 0

                ats = compute_ats_score(skill_match_ratio, keyword_coverage, similarity_pct)

                rows.append({
                    "candidate_id": cid,
                    "resume_text": doc_text,
                    "similarity_pct": similarity_pct,
                    "matched_skills": matched,
                    "missing_skills": missing,
                    "ats_score": ats,
                    "snippet_html": highlight_keywords(doc_text, jd_skills, 900)
                })

        rows_sorted = sorted(rows, key=lambda x: x["ats_score"], reverse=True)
        display = rows_sorted[:top_k]

        st.markdown("<h3 class='section-title'>3️⃣ Ranked Candidates</h3>", unsafe_allow_html=True)

        for i, r in enumerate(display):
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.markdown(
                f"### 🏆 Rank #{i+1}: <b>{r['candidate_id']}</b> "
                f"<span style='color:#2457A5;'> (ATS Score: {r['ats_score']}/100)</span>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.write(f"**Similarity:** {r['similarity_pct']}%")
                st.write(f"**Matched Skills:** {', '.join(r['matched_skills']) or 'None'}")
                st.write(f"**Missing Skills:** {', '.join(r['missing_skills']) or 'None'}")

            with col2:
                if show_full_resume:
                    st.text_area("Full Resume Text", r["resume_text"], height=240)
                else:
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(255,255,255,0.6);
                            padding: 15px;
                            border-radius: 12px;
                            border-left: 5px solid #5B8DEF;
                            box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
                        ">
                            {r["snippet_html"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.download_button(
                f"⬇️ Download {r['candidate_id']} (TXT)",
                r["resume_text"],
                file_name=f"{r['candidate_id']}.txt",
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # CSV Export
        df = pd.DataFrame([
            {
                "candidate_id": r["candidate_id"],
                "ats_score": r["ats_score"],
                "similarity_pct": r["similarity_pct"],
                "matched_skills": ", ".join(r["matched_skills"]),
                "missing_skills": ", ".join(r["missing_skills"]),
            }
            for r in rows_sorted
        ])

        st.download_button(
            "📥 Download Results CSV",
            df.to_csv(index=False),
            file_name="ranked_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.success("✔ Screening Complete!")

