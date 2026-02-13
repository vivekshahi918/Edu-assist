import streamlit as st
import google.generativeai as genai

import PyPDF2 as pdf
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Get available models and select the best one
models = genai.list_models()
available_models = [model.name for model in models]
print("Available models for ATS:", available_models)

# Define preferred models in order of preference
preferred_models = ["models/gemini-2.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision", 
                   "models/gemini-pro", "models/gemini-1.0-pro"]

# Select the first available preferred model
model_name = None
for name in preferred_models:
    if name in available_models or (name.startswith("models/") and 
                                   name[7:] in [m[7:] if m.startswith("models/") else m for m in available_models]):
        model_name = name
        break

# Fallback to any available model if preferred models aren't available
if not model_name and available_models:
    for model in available_models:
        if "gemini" in model.lower() and not model.endswith("vision"):
            model_name = model
            break
    
    # If still no model found, use the first available
    if not model_name and available_models:
        model_name = available_models[0]

print(f"Using model for ATS: {model_name}")
model = genai.GenerativeModel(model_name)

def extract_json_from_text(text):
    """Extract JSON from text that might contain extra content."""
    # Try to find JSON in the text using regex
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If the extracted text isn't valid JSON, try cleaning it further
            pass
    
    # If regex didn't work, try finding JSON brackets manually
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    
    if start_idx >= 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If still not valid JSON, return None
            return None
    
    return None

def analyze_resume(resume_text, job_description, job_role):
    """Analyze resume against job description using AI."""
    
    # Limit text length to avoid token limits
    max_length = 10000
    if len(resume_text) > max_length:
        resume_text = resume_text[:max_length] + "..."
    if len(job_description) > max_length:
        job_description = job_description[:max_length] + "..."
    
    input_prompt = f'''
    You're an advanced ATS (Applicant Tracking System) expert analyzing resumes.
    
    TASK:
    Analyze the provided resume against the job description for the role of {job_role}.
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    PROVIDE YOUR ANALYSIS IN THE FOLLOWING JSON FORMAT:
    {{
        "PercentageMatch": "XX%",
        "MissingKeywordsintheResume": ["keyword1", "keyword2"],
        "FoundKeywords": ["keyword1", "keyword2"],
        "KeySkillGaps": ["skill1", "skill2"],
        "ResumeImprovementSuggestions": ["suggestion1", "suggestion2"],
        "ProfileSummary": "Brief assessment",
        "StrengthsForRole": ["strength1", "strength2"],
        "InterviewTips": ["tip1", "tip2"]
    }}
    
    IMPORTANT: RESPOND ONLY WITH THE PROPERLY FORMATTED JSON. DO NOT ADD ANY TEXT BEFORE OR AFTER THE JSON.
    '''

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    try:
        # Generate content with safety settings and higher temperature for more creativity
        response = model.generate_content(
            input_prompt,
            generation_config={"temperature": 0.2, "top_p": 0.95, "top_k": 40},
            safety_settings=safety_settings
        )
        
        # For debugging
        st.write("Response received. Processing...")
        
        # Get response text
        if hasattr(response, 'text'):
            response_text = response.text
        else:
            # Handle different response formats
            try:
                response_text = response.candidates[0].content.parts[0].text
            except:
                response_text = str(response)
        
        # For debugging - show the raw response
        st.expander("View Raw AI Response").code(response_text)
        
        # Try to extract and parse JSON from response
        json_data = extract_json_from_text(response_text)
        
        # If we couldn't parse JSON, create a fallback response
        if not json_data:
            st.warning("Could not parse JSON response. Using fallback analysis.")
            
            # Create a fallback analysis with basic information
            keywords = extract_keywords(job_description)
            found = [kw for kw in keywords if kw.lower() in resume_text.lower()]
            missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
            
            # Calculate a simple match percentage
            match_percentage = int((len(found) / max(len(keywords), 1)) * 100) if keywords else 50
            match_percentage = f"{match_percentage}%"
            
            json_data = {
                "PercentageMatch": match_percentage,
                "MissingKeywordsintheResume": missing[:5],  # Limit to top 5
                "FoundKeywords": found[:5],  # Limit to top 5
                "KeySkillGaps": missing[:3],  # Use missing keywords as skill gaps
                "ResumeImprovementSuggestions": [
                    "Add more keywords relevant to the job description",
                    "Highlight your relevant experience more clearly",
                    "Quantify your achievements with metrics"
                ],
                "ProfileSummary": "We were unable to generate a detailed profile summary. Please try again with a different resume or job description.",
                "StrengthsForRole": found[:3],  # Use found keywords as strengths
                "InterviewTips": [
                    "Research the company before the interview",
                    "Prepare examples of your past experiences",
                    "Practice answering common interview questions"
                ]
            }
        
        return json_data
        
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        st.warning("Using fallback analysis method")
        
        # Create a simple fallback analysis
        return create_fallback_analysis(resume_text, job_description)

def extract_keywords(text):
    """Extract potential keywords from job description."""
    # Common job-related keywords to look for
    common_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "spring", "hibernate", "sql",
        "nosql", "mongodb", "postgresql", "mysql", "oracle", "aws", "azure",
        "gcp", "docker", "kubernetes", "ci/cd", "git", "agile", "scrum",
        "leadership", "communication", "teamwork", "problem-solving", "analytics",
        "data science", "machine learning", "ai", "security", "devops", "testing"
    ]
    
    # Find all words that might be skills or technologies
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9+#\-.]*[A-Za-z0-9]\b', text.lower())
    
    # Filter to keep only common skills and capitalize them properly
    keywords = [word for word in words if word in common_skills]
    
    # Deduplicate
    return list(set(keywords))

def create_fallback_analysis(resume_text, job_description):
    """Create a basic analysis when the AI model fails."""
    keywords = extract_keywords(job_description)
    found = [kw for kw in keywords if kw.lower() in resume_text.lower()]
    missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
    
    # Calculate a simple match percentage
    match_percentage = int((len(found) / max(len(keywords), 1)) * 100) if keywords else 50
    match_percentage = f"{match_percentage}%"
    
    return {
        "PercentageMatch": match_percentage,
        "MissingKeywordsintheResume": missing[:5],  # Limit to top 5
        "FoundKeywords": found[:5],  # Limit to top 5
        "KeySkillGaps": missing[:3],  # Use missing keywords as skill gaps
        "ResumeImprovementSuggestions": [
            "Add more keywords relevant to the job description",
            "Highlight your relevant experience more clearly",
            "Quantify your achievements with metrics"
        ],
        "ProfileSummary": "This is an automated analysis. The resume was compared to the job description and key skills were identified. Consider adding the missing keywords to improve your match score.",
        "StrengthsForRole": found[:3] if found else ["Experience", "Education", "Skills"],  # Use found keywords as strengths
        "InterviewTips": [
            "Research the company before the interview",
            "Prepare examples of your past experiences",
            "Practice answering common interview questions"
        ]
    }

def create_match_radar_chart(data):
    """Generate radar chart showing skills match."""
    categories = ['Technical Skills', 'Experience', 'Education', 
                  'Communication', 'Cultural Fit']
    
    # Generate match percentages based on the overall match
    base_match = int(data['PercentageMatch'].strip('%'))
    # Create slightly varied values for different categories
    values = [
        min(100, base_match + np.random.randint(-10, 15)),  # Technical Skills
        min(100, base_match + np.random.randint(-8, 12)),   # Experience
        min(100, base_match + np.random.randint(-5, 10)),   # Education
        min(100, base_match + np.random.randint(-12, 8)),   # Communication
        min(100, base_match + np.random.randint(-7, 10))    # Cultural Fit
    ]
    
    # Make sure none are below 0
    values = [max(0, v) for v in values]
    
    # Create radar chart
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    values += values[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Add value labels
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value + 5, f'{value}%', 
                ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Set y-axis limit
    ax.set_ylim(0, 100)
    plt.title('Skills Match Analysis', size=15, fontweight='bold', pad=15)
    
    return fig

def main():
    st.write("<h1><center>Advanced ATS Analyzer</center></h1>", unsafe_allow_html=True)
    st.text("👉🏻                  Personal ATS for Job-Seekers & Recruiters                   👈")
    
    # Load the animation
    try:
        with open('src/ATS.json') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 200, -200)
    except Exception as e:
        st.warning(f"Animation file not found or couldn't be loaded: {str(e)}")
    
    # Two-column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Job role input
        job_role = st.text_input("Job Role", placeholder="e.g. Senior Python Developer")
        
        # File upload for resume
        uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload PDF file only")
    
    with col2:
        # Job description input
        job_desc = st.text_area("Paste the Job Description", height=200)
    
    # Submit button
    submit = st.button("Analyze Resume")

    if submit:
        if uploaded_file is not None and job_desc and job_role:
            # Reading the uploaded PDF file
            try:
                reader = pdf.PdfReader(uploaded_file)
                text = ""
                for page_number in range(len(reader.pages)):
                    page = reader.pages[page_number]
                    text += str(page.extract_text())
                
                # Spinner while evaluating
                with st.spinner("Analyzing your resume against job requirements..."):
                    response_data = analyze_resume(text, job_desc, job_role)
                
                if response_data:
                    # Display the ATS scanner results in tabs
                    tab1, tab2, tab3 = st.tabs(["Match Analysis", "Skills Gap", "Improvement Tips"])
                    
                    with tab1:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Match Score")
                            # Create a large, centered percentage display
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <h1 style="font-size: 4rem; color: {'green' if int(response_data['PercentageMatch'].strip('%')) >= 70 else 'orange' if int(response_data['PercentageMatch'].strip('%')) >= 50 else 'red'};">
                                    {response_data['PercentageMatch']}
                                </h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("Profile Summary")
                            st.write(response_data['ProfileSummary'])
                        
                        with col2:
                            # Display radar chart
                            st.subheader("Skills Match Analysis")
                            fig = create_match_radar_chart(response_data)
                            st.pyplot(fig)
                    
                    with tab2:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Missing Keywords")
                            if response_data['MissingKeywordsintheResume']:
                                for keyword in response_data['MissingKeywordsintheResume']:
                                    st.markdown(f"- 🔴 {keyword}")
                            else:
                                st.success("No missing keywords found!")
                        
                        with col2:
                            st.subheader("Found Keywords")
                            if response_data['FoundKeywords']:
                                for keyword in response_data['FoundKeywords']:
                                    st.markdown(f"- ✅ {keyword}")
                            else:
                                st.warning("No matching keywords found")
                        
                        st.subheader("Key Skill Gaps")
                        if response_data['KeySkillGaps']:
                            for skill in response_data['KeySkillGaps']:
                                st.markdown(f"- 🔍 {skill}")
                        else:
                            st.success("No major skill gaps identified!")
                            
                        st.subheader("Your Strengths")
                        if response_data['StrengthsForRole']:
                            for strength in response_data['StrengthsForRole']:
                                st.markdown(f"- 💪 {strength}")
                    
                    with tab3:
                        st.subheader("Resume Improvement Suggestions")
                        if response_data['ResumeImprovementSuggestions']:
                            for i, suggestion in enumerate(response_data['ResumeImprovementSuggestions'], 1):
                                st.markdown(f"**{i}.** {suggestion}")
                        else:
                            st.success("Your resume looks great!")
                        
                        st.subheader("Interview Preparation Tips")
                        if response_data['InterviewTips']:
                            for i, tip in enumerate(response_data['InterviewTips'], 1):
                                st.markdown(f"**{i}.** {tip}")
                
                    # Add download button for a summary report
                    missing_keywords = (
                        "- " + "\n- ".join(response_data['MissingKeywordsintheResume'])
                        if response_data['MissingKeywordsintheResume']
                        else "No missing keywords found!"
                    )

                    skill_gaps = (
                        "- " + "\n- ".join(response_data['KeySkillGaps'])
                        if response_data['KeySkillGaps']
                        else "No major skill gaps identified!"
                    )

                    improvements = (
                        "1. " + "\n2. ".join(response_data['ResumeImprovementSuggestions'])
                        if response_data['ResumeImprovementSuggestions']
                        else "Your resume looks great!"
                    )

                    interview_tips = (
                        "1. " + "\n2. ".join(response_data['InterviewTips'])
                        if response_data['InterviewTips']
                        else "No tips available."
                    )

                    report = f"""# Resume Analysis Report for {job_role}

                    ## Overall Match: {response_data['PercentageMatch']}

                    ### Profile Summary
                    {response_data['ProfileSummary']}

                    ### Missing Keywords
                    {missing_keywords}

                    ### Key Skill Gaps
                    {skill_gaps}

                    ### Resume Improvement Suggestions
                    {improvements}

                    ### Interview Tips
                    {interview_tips}
                    """

                    
                    st.download_button(
                        label="Download Analysis Report",
                        data=report,
                        file_name="resume_analysis_report.md",
                        mime="text/markdown"
                    )
                
                else:
                    st.error("Failed to analyze resume. Please try again.")
            
            except Exception as e:
                st.error(f"Error processing your resume: {str(e)}")
        else:
            st.warning("Please provide a job role, upload your resume, and paste the job description.")

if __name__ == "__main__":
    main()
