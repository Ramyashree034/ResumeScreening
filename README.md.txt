AI-Powered Resume Screening Agent

A smart AI assistant that automatically parses resumes, extracts skills, matches them to job descriptions, and returns ranked candidates — all using Streamlit and OpenAI.

⭐ Overview

This project is an AI-powered resume screening application.
Given a job description, the system:

Extracts text from resumes (PDF or DOCX)

Uses AI embeddings to match resumes with the job description

Ranks resumes based on relevance

Displays matched candidates with match percentage

Allows uploading multiple resumes at once

Highlights recommended candidates

This tool helps HR teams or recruiters automate the first-level resume filtering.

🚀 Features
✅ Multiple Resume Upload (PDF + DOCX)
✅ AI-Based Resume–Job Match Ranking
✅ Skill Extraction using Embeddings
✅ Clean UI with Streamlit
✅ Works fully locally (no databases needed)
✅ Supports future improvements (LLM explanations, skill graphs, etc.)
🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
AI Model	OpenAI Embeddings (text-embedding-3-small)
Parsing	PyPDF2, python-docx
Language	Python 3.10+