import streamlit as st
import os
import json
# Import the functions and schemas we engineered in app.py
from app import extract_text_from_pdf, match_resume_to_job

# Set up clean page configurations
st.set_page_config(page_title="Stark ATS Analyzer", page_icon="🎯", layout="wide")

st.title("🎯 Enterprise AI Resume Matcher & ATS Engine")
st.caption("Powered by Gemini 2.5 Flash & Semantic Data Mapping")
st.markdown("---")

# Create a clean side-by-side two-column workspace
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Step 1: Input Requirements")
    # Interactive text box for pasting target job requirements
    job_description = st.text_area(
        "Paste the Job Description (JD) here:",
        height=250,
        placeholder="Looking for a Full-Stack Developer with React, Node.js, and MongoDB experience..."
    )
    
    # Drag-and-drop file uploader block for the candidate's PDF
    uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])

with col2:
    st.subheader("📊 Step 2: System Evaluation Insights")
    
    # Evaluation execution trigger button
    if st.button("Run Semantic ATS Analysis", type="primary"):
        if not job_description:
            st.error("Missing Parameter: Please paste a Job Description first.")
        elif not uploaded_file:
            st.error("Missing Parameter: Please upload a Resume PDF file.")
        else:
            with st.spinner("Processing binary stream and generating AI metrics..."):
                try:
                    # 1. Save uploaded file buffer to a temporary file locally
                    temp_filename = "temp_uploaded_resume.pdf"
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 2. Extract text layout
                    raw_resume_text = extract_text_from_pdf(temp_filename)
                    
                    # 3. Call our AI matching core
                    raw_json_response = match_resume_to_job(raw_resume_text, job_description)
                    
                    # Parse the raw string response back into a clean Python dictionary
                    data = json.loads(raw_json_response)
                    
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    
                    # 4. Render the Metrics beautifully in the UI
                    st.success("Analysis Complete!")
                    
                    # Display Candidate Name and Big Score Metric
                    st.metric(label="Match Alignment Score", value=f"{data.get('match_percentage', 0)}%")
                    st.markdown(f"**Candidate Evaluated:** {data.get('candidate_name', 'Unknown')}")
                    
                    # Display Experience Fit Analysis Block
                    st.info(data.get('experience_fit_analysis', 'No structural analysis provided.'))
                    
                    # Side by side breakdown of matched vs missing skills
                    sk_col1, sk_col2 = st.columns(2)
                    with sk_col1:
                        st.markdown("### ✅ Matched Competencies")
                        for skill in data.get('matched_skills', []):
                            st.markdown(f"- {skill}")
                            
                    with sk_col2:
                        st.markdown("### ❌ Identified Skill Gaps")
                        for skill in data.get('missing_critical_skills', []):
                            st.markdown(f"- <span style='color:#ef5350'>{skill}</span>", unsafe_allow_html=True)
                    
                    # Render actionable bullet points at the bottom
                    st.markdown("---")
                    st.markdown("### 🛠️ Strategic Optimization Feedback")
                    for bullet in data.get('actionable_feedback', []):
                        st.markdown(f"- {bullet}")
                        
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")