from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import re
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import time
from streamlit_lottie import st_lottie
import json
import PyPDF2
import docx2txt
from PIL import Image
import io
import base64
import httpx

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Constants for prompts
DETAILED_PROMPT = """
You are an expert educational content summarizer. Create comprehensive notes including:
1. TITLE: Create an appropriate title based on content
2. SUMMARY: A concise 2-3 sentence summary
3. KEY CONCEPTS: 4-6 main ideas with explanations
4. DETAILED NOTES: Organized structure with headings and bullet points
5. IMPORTANT TERMINOLOGY: List and define key terms
6. APPLICATION: 2-3 ways to apply this information
7. CONNECTIONS: How this connects to related topics

Format using proper Markdown with headings, bullet points, emphasis, and code blocks as needed.
"""

SIMPLE_PROMPT = """
Summarize this content into concise notes with:
- Key points and main takeaways
- Bullet points for easy reading
- Under 250 words total
"""

def load_animation():
    """Load the Lottie animation file if available"""
    try:
        with open('src/Notes.json', encoding='utf-8') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 100, -200)
    except Exception as e:
        st.info("Note: Animation not loaded. This doesn't affect functionality.")

def select_model():
    """Select the best available Gemini model"""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = genai.list_models()
    available_models = [model.name for model in models]
    
    # Try models in order of preference
    for name in ["models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-pro"]:
        if name in available_models or (name.startswith("models/") and 
                                      name[7:] in [m[7:] if m.startswith("models/") else m for m in available_models]):
            print(f"Using model: {name}")
            return genai.GenerativeModel(name)
    
    # Fallback to any available model
    if available_models:
        print(f"Using fallback model: {available_models[0]}")
        return genai.GenerativeModel(available_models[0])
    else:
        st.error("No available models found. Please check your API key.")
        return None

def extract_video_id(youtube_url):
    """Extract video ID from various YouTube URL formats"""
    if not youtube_url:
        return None
    
    # Standard URL format
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    
    # Short URL format
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    # Embedded format
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com') and parsed_url.path.startswith('/embed/'):
        return parsed_url.path.split('/')[2]
    
    # Direct ID input
    if re.match(r'^[A-Za-z0-9_-]{11}$', youtube_url):
        return youtube_url
    
    # Regex fallback
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    return match.group(1) if match else None

def get_video_details(video_id):
    """Get video title and thumbnail URL"""
    title = "YouTube Video"
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.replace(' - YouTube', '')
    except Exception:
        pass
    
    return {
        "title": title,
        "thumbnail": f"http://img.youtube.com/vi/{video_id}/0.jpg"
    }

def get_transcript_from_youtube_api(video_id):
    """Try to get transcript using the YouTube Transcript API directly"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception:
        return None

def get_transcript_from_alternative_apis(video_id):
    """Try alternative APIs to get YouTube transcripts"""
    # Option 1: Try using a public transcript API
    try:
        response = httpx.get(
            f"https://yt-downloader-six.vercel.app/transcript?id={video_id}",
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("transcript"):
                return " ".join(item["text"] for item in data["transcript"])
    except Exception:
        pass
    
    # Option 2: Try using the Rapid API (requires API key setup in .env)
    rapid_api_key = os.getenv("RAPID_API_KEY")
    if rapid_api_key:
        try:
            url = "https://youtube-transcriptor.p.rapidapi.com/transcript"
            headers = {
                "X-RapidAPI-Key": rapid_api_key,
                "X-RapidAPI-Host": "youtube-transcriptor.p.rapidapi.com"
            }
            params = {"video_id": video_id}
            response = requests.get(url, headers=headers, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                transcript = data.get("transcript", "")
                if transcript:
                    return transcript
        except Exception:
            pass
    
    return None

def get_transcript_from_invidious(video_id):
    """Try to get transcript from Invidious API (open source YouTube alternative)"""
    try:
        invidious_instances = [
            "https://invidious.snopyta.org",
            "https://invidious.kavin.rocks",
            "https://yewtu.be"
        ]
        
        for instance in invidious_instances:
            try:
                url = f"{instance}/api/v1/captions/{video_id}"
                response = requests.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        caption_url = None
                        for caption in data:
                            if caption.get("languageCode") == "en":
                                caption_url = caption.get("url")
                                break
                        
                        if not caption_url and len(data) > 0:
                            caption_url = data[0].get("url")
                            
                        if caption_url:
                            caption_response = requests.get(caption_url, timeout=5.0)
                            if caption_response.status_code == 200:
                                # Process and extract text from the caption response
                                lines = caption_response.text.strip().split('\n')
                                transcript = ""
                                for i in range(0, len(lines), 4):
                                    if i+2 < len(lines):
                                        transcript += lines[i+2] + " "
                                return transcript
            except Exception:
                continue
                
    except Exception:
        pass
        
    return None

def get_transcript(video_id):
    """Get transcript using multiple methods with fallbacks"""
    # Try YouTube API first (most reliable when it works)
    transcript = get_transcript_from_youtube_api(video_id)
    if transcript:
        return transcript
        
    # Try alternative APIs
    transcript = get_transcript_from_alternative_apis(video_id)
    if transcript:
        return transcript
        
    # Try Invidious API
    transcript = get_transcript_from_invidious(video_id)
    if transcript:
        return transcript
    
    # If all methods fail, raise exception
    raise Exception("Could not retrieve transcript using any available method. Try another video or upload a file with content instead.")

def extract_text_from_file(uploaded_file):
    """Extract text from various file types"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        # PDF Files
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
            
        # Word Documents
        elif file_extension in ['docx', 'doc']:
            text = docx2txt.process(uploaded_file)
            return text
            
        # Text files
        elif file_extension in ['txt', 'md', 'markdown', 'csv']:
            return uploaded_file.getvalue().decode('utf-8')
            
        # Images (using OCR)
        elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
            if PYTESSERACT_AVAILABLE:
                try:
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image)
                    return text
                except Exception as e:
                    st.error(f"OCR Error: {str(e)}.")
                    st.info("OCR processing is not working. Try uploading a text-based file instead.")
                    return None
            else:
                st.error("Image text extraction is not available in this deployment.")
                st.info("OCR functionality requires pytesseract which is not installed. Please upload a text-based file instead.")
                return None
                
        # Unsupported file type
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return None

def generate_notes(model, text, is_detailed=True):
    """Generate notes from text content using AI model"""
    prompt = DETAILED_PROMPT if is_detailed else SIMPLE_PROMPT
    
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        # Limit text length to avoid token limits
        max_tokens = 25000
        if len(text) > max_tokens:
            text = text[:max_tokens] + "...[text truncated due to length]"
        
        response = model.generate_content(
            prompt + "\n\nCONTENT:\n" + text,
            generation_config={"temperature": 0.2, "top_k": 40, "top_p": 0.95},
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating notes: {str(e)}")
        # If detailed notes fail, try simpler ones
        if is_detailed:
            st.info("Trying simplified notes generation...")
            return generate_notes(model, text[:5000], False)
        return None

def main():
    st.write("<h1><center>Advanced Notes Generator</center></h1>", unsafe_allow_html=True)
    
    # Load animation and initialize model
    load_animation()
    model = select_model()
    
    if not model:
        st.error("Failed to initialize AI model. Please check your API key.")
        return
    
    # UI components
    st.subheader("Transform Content into Comprehensive Study Notes")
    
    # Create tabs for YouTube and File Upload
    tab1, tab2 = st.tabs(["YouTube Video", "Upload File"])
    
    with tab1:
        st.markdown("### Generate Notes from YouTube Video")
        youtube_link = st.text_input("Enter YouTube video link", help="Paste the full YouTube URL", key="youtube_input")
        
        col1, col2 = st.columns(2)
        with col1:
            note_style_yt = st.selectbox("Note Style", ["Comprehensive", "Summary"], key="note_style_yt")
        
        if st.button("Generate Notes from Video", key="generate_yt"):
            if not youtube_link:
                st.warning("Please enter a YouTube URL")
            else:
                # Process video
                with st.spinner("Processing YouTube video..."):
                    try:
                        video_id = extract_video_id(youtube_link)
                        if not video_id:
                            st.error("Invalid YouTube URL format. Please check and try again.")
                        else:
                            # Get video details
                            video = get_video_details(video_id)
                            st.markdown(f"### {video['title']}")
                            st.image(video['thumbnail'], use_container_width=True)
                            
                            # Try to get transcript
                            with st.spinner("Fetching video transcript..."):
                                try:
                                    transcript = get_transcript(video_id)
                                    if not transcript:
                                        st.error("Failed to retrieve transcript. Try uploading a file instead.")
                                    else:
                                        st.success(f"Transcript obtained ({len(transcript.split())} words)")
                                        
                                        # Generate notes
                                        with st.spinner("Generating notes..."):
                                            notes = generate_notes(model, transcript, note_style_yt == "Comprehensive")
                                            
                                            if notes:
                                                display_notes(notes, video['title'])
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.info("Tip: If transcript retrieval fails, try uploading a file with the content instead.")
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
    
    with tab2:
        st.markdown("### Generate Notes from Uploaded File")
        
        # Update file types based on what's actually supported in the deployment
        supported_types = ["pdf", "docx", "doc", "txt", "md"]
        if PYTESSERACT_AVAILABLE:
            supported_types.extend(["jpg", "jpeg", "png"])
            upload_message = "Upload a document (PDF, DOCX, TXT) or image"
        else:
            upload_message = "Upload a document (PDF, DOCX, TXT)"
        
        uploaded_file = st.file_uploader(upload_message, type=supported_types)
        
        col1, col2 = st.columns(2)
        with col1:
            note_style_file = st.selectbox("Note Style", ["Comprehensive", "Summary"], key="note_style_file")
        
        if st.button("Generate Notes from File", key="generate_file"):
            if uploaded_file is None:
                st.warning("Please upload a file first")
            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Extract text from the file
                        text = extract_text_from_file(uploaded_file)
                        
                        if text:
                            st.success(f"Text extracted: {len(text.split())} words")
                            
                            # Generate notes
                            with st.spinner("Generating notes..."):
                                notes = generate_notes(model, text, note_style_file == "Comprehensive")
                                
                                if notes:
                                    display_notes(notes, uploaded_file.name)
                        else:
                            st.error("Could not extract text from the uploaded file.")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")

def display_notes(notes, title="Content"):
    """Display the generated notes with a download button"""
    st.markdown("## Generated Notes")
    
    with st.expander("View Full Notes", expanded=True):
        st.markdown(notes)
    
    # Create a safe filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    
    # Add download button
    st.download_button(
        label="Download Notes",
        data=notes,
        file_name=f"{safe_title[:50]}_notes.md",
        mime="text/markdown"
    )
    
    # Add copy button
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 15px 0;">
        <button 
            onclick="
                navigator.clipboard.writeText(document.querySelector('.stMarkdown').innerText);
                this.innerText='Copied!';
                setTimeout(() => this.innerText='Copy to Clipboard', 2000)
            "
            style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;"
        >
            Copy to Clipboard
        </button>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
