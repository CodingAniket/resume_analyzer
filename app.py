import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import pdfplumber 

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize the Gemini Client
client = genai.Client()

# 3. UPGRADED: New Pydantic schema for deep ATS Match Analysis
class JobMatchAnalysis(BaseModel):
    candidate_name: str = Field(description="The full name of the candidate")
    match_percentage: int = Field(description="A score from 0 to 100 indicating alignment with the job description")
    matched_skills: List[str] = Field(description="Skills the candidate possesses that directly align with the job requirements")
    missing_critical_skills: List[str] = Field(description="Crucial skills, frameworks, or concepts mentioned in the JD that are completely missing from the resume")
    experience_fit_analysis: str = Field(description="A concise logical analysis explaining if the candidate's project depth and experience match the seniority of the role")
    actionable_feedback: List[str] = Field(description="Direct, constructive bullet points detailing exactly how the candidate can rewrite or improve their resume for this role")

# Helper function to extract text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text

# 4. UPGRADED: Core AI Matching Function
def match_resume_to_job(resume_text: str, job_description: str) -> str:
    prompt = f"""
    You are a senior technical recruiter at Stark Industries. Your task is to critically analyze the provided Resume against the Job Description (JD).
    
    Perform a deep semantic analysis:
    1. Look past exact keywords—identify underlying concepts and technical frameworks (e.g., if JD asks for 'MERN stack' and resume lists 'MongoDB, Express, React, Node', that is a valid match).
    2. Evaluate the complexity of projects and engineering fundamentals against what the JD requires.
    3. Identify critical missing tech stacks or methodologies.
    4. Provide honest, logical, and highly actionable feedback to improve the resume.

    Job Description:
    {job_description}

    Candidate Resume:
    {resume_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobMatchAnalysis, # Enforcing our new multi-dimensional schema
            temperature=0.2 # Kept low for consistent, logical evaluation
        ),
    )
    return response.text

# 5. Execution Block
if __name__ == "__main__":
    # Make sure you have your target PDF named 'resume.pdf' in the folder
    pdf_filename = "resume.pdf" 
    
    
    if os.path.exists(pdf_filename):
        print(f"1. Extracting text from binary stream: {pdf_filename}...")
        raw_resume_text = extract_text_from_pdf(pdf_filename)
        
        print("2. Spawning Gemini engine for semantic ATS evaluation...")
        analysis_result = match_resume_to_job(raw_resume_text, target_job_description)
        
        print("\n================== ATS MATCH ANALYZER INSIGHTS ==================")
        print(analysis_result)
        print("=================================================================")
    else:
        print(f"Error: Please place your PDF named '{pdf_filename}' in this folder to run the matcher.")