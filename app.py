import streamlit as st

from google import generativeai as genai

genai.configure(api_key="AIzaSyBoM04atPjgLw6ck00pd6xzh9vxGHvA9LQ")

from st_on_hover_tabs import on_hover_tabs
import json
from streamlit_lottie import st_lottie
from menu.mcqgen import main as mcq
from menu.NotesMaker import main as notes
from menu.Contest_Calendar import main as contest_calendar
from menu.Ask_To_PDF import main as ask_to_pdf_page
from menu.ATS import main as ats_page

# Setting the page configuration
st.set_page_config(page_title="EduAssist", page_icon='src/Logo College.png', layout='wide')

# Initialize session state for theme
if "current_theme" not in st.session_state:
    st.session_state.current_theme = "light"

st.markdown('<style>' + open('./src/style.css').read() + '</style>', unsafe_allow_html=True)

def home():
    st.markdown("<h1 style='text-align: center;'>Welcome to EduAssist!</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>AI-powered System for Students</h4>", unsafe_allow_html=True)

    try:
        with open('src/Home_student.json', encoding='utf-8') as anim_source:
            animation_data = json.load(anim_source)
        st_lottie(animation_data, 1, True, True, "high", 350, -200)
    except FileNotFoundError:
        st.error("Animation file not found.")
    except UnicodeDecodeError as e:
        st.error(f"Error decoding JSON: {e}. Try specifying a different encoding.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)

def main():
    st.markdown("""
        <style>
            /* Reduce padding for the entire page */
            .css-1y0tads, .block-container, .css-1lcbmhc {
                padding-top: 0px !important;
                padding-bottom: 0px !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Display theme change button within the sidebar
        st.image('src/Logo College.png', width=70)
       
        tabs = on_hover_tabs(
            tabName=['Home', 'MCQ Generator','Ask To PDF', 'Notes Maker','Contest Calendar','ATS'], 
            iconName=['home', 'center_focus_weak','search', 'edit','calendar_month','work'], 
            default_choice=0
        )

    menu = {
        'Home': home,
        'MCQ Generator': mcq,
        'Ask To PDF': ask_to_pdf_page,
        'Notes Maker': notes,
        'Contest Calendar':contest_calendar,
        'ATS': ats_page,
    }
    
    menu[tabs]()

if __name__ == "__main__":
    main()

