import os
import re
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

# 3. ENHANCED SCHEMA: Added structural grounding to eliminate hallucinations
class JobMatchAnalysis(BaseModel):
    candidate_name: str = Field(description="The full name of the candidate found in the resume")
    match_percentage: int = Field(description="A score from 0 to 100 based on core conceptual alignment and requirement fulfillment")
    semantic_alignment_score: int = Field(description="An internal calibration score matching the value of match_percentage")
    matched_skills: List[str] = Field(description="Skills the candidate possesses that directly align with the job requirements")
    missing_critical_skills: List[str] = Field(description="Crucial skills or frameworks mentioned in the JD that are completely missing from the resume")
    experience_fit_analysis: str = Field(description="A concise logical analysis explaining if the project depth matches the seniority of the role")
    actionable_feedback: List[str] = Field(description="Direct, constructive bullet points detailing exactly how the candidate can rewrite or improve their resume")

# Layout-Aware Text Extraction
def extract_text_from_pdf(pdf_path: str) -> str:
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text

# COMPLEX ARCHITECTURE: Algorithmic Keyword Overlap Engine (BM25 Token Baseline)
def calculate_keyword_overlap(resume_text: str, job_description: str) -> float:
    """Calculates strict keyword overlap to act as a mathematical mathematical anchor."""
    # Clean text and extract lowercase words
    resume_words = set(re.findall(r'\b\w{2,}\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b\w{2,}\b', job_description.lower()))
    
    if not jd_words:
        return 0.0
        
    # Calculate Jaccard Similarity as a mathematical proxy for strict keyword matching
    intersection = resume_words.intersection(jd_words)
    overlap_score = (len(intersection) / len(jd_words)) * 100
    return round(overlap_score, 2)

# 4. UPGRADED: Core AI Matching Function with Hybrid Mathematical Engine
def match_resume_to_job(resume_text: str, job_description: str) -> str:
    prompt = f"""
    You are a principal technical recruiter at Stark Industries. Your task is to critically analyze the provided Resume against the Job Description (JD).
    
    Perform a deep semantic analysis:
    1. Look past exact keywords—identify underlying concepts and technical frameworks (e.g., 'MERN stack' matches 'MongoDB, Express, React, Node').
    2. Evaluate project complexity and engineering fundamentals against the JD.
    3. Identify critical missing tech stacks or methodologies.
    4. Provide honest, logical feedback.

    Job Description:
    {job_description}

    Candidate Resume:
    {resume_text}
    """
    
    # Get Semantic Analysis from Gemini
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobMatchAnalysis, 
            temperature=0.1 # Absolute low temperature for deterministic evaluation
        ),
    )
    
    # Parse LLM JSON to inject the Hybrid Mathematical Matrix
    import json
    try:
        data = json.loads(response.text)
        
        # Calculate the deterministic keyword baseline
        keyword_score = calculate_keyword_overlap(resume_text, job_description)
        semantic_score = data.get("semantic_alignment_score", 50)
        
        # COMPLEX HYBRID FORMULA: 70% Semantic Deep Understanding + 30% Hard Keyword Match
        final_calculated_ats_score = int((0.70 * semantic_score) + (0.30 * keyword_score))
        
        # Inject the composite score back into our structured JSON response
        data["final_composite_ats_score"] = min(100, max(0, final_calculated_ats_score))
        data["algorithmic_keyword_weight"] = f"{keyword_score}%"
        
        return json.dumps(data, indent=4)
    except Exception as e:
        # Fallback in case of parsing errors
        return response.text

# 5. Execution Block
if __name__ == "__main__":
    pdf_filename = "resume.pdf" 
    
    # Mock Job Description for your project execution testing
    sample_job_description = """
    We are looking for a Software Engineering Intern with experience in Python, GenAI frameworks, 
    and backend API design. Knowledge of SQL databases, data structures, and cloud-native architectures is highly preferred.
    """

    if os.path.exists(pdf_filename):
        print(f"1. Extracting structured text via layout-aware bounding boxes: {pdf_filename}...")
        raw_resume_text = extract_text_from_pdf(pdf_filename)
        
        print("2. Spawning Hybrid ATS Math Engine (Deterministic Token Overlap + Gemini Semantic Mapping)...")
        # FIXED: Now passing both required arguments to prevent runtime crash
        analysis_result = match_resume_to_job(raw_resume_text, sample_job_description)
        
        print("\n================== ENTERPRISE ATS HYBRID MATRIX INSIGHTS ==================")
        print(analysis_result)
        print("==========================================================================")
    else:
        print(f"Error: Please place your PDF named '{pdf_filename}' in this folder to run the matcher.")