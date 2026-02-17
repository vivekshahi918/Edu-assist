import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_lottie import st_lottie 
import json
import faiss
import pickle
import asyncio
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# List available models for debugging
models = genai.list_models()
available_models = [model.name for model in models]
print("Available models:", available_models)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    faiss.write_index(vector_store.index, "faiss_index.bin")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump({"docstore": vector_store.docstore, "index_to_docstore_id": vector_store.index_to_docstore_id}, f)

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_store.pkl", "rb") as f:
        store_data = pickle.load(f)
    vector_store = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=store_data["docstore"], index_to_docstore_id=store_data["index_to_docstore_id"])
    return vector_store

async def get_conversational_chain():
    prompt_template = """
    Leave First 1 line empty and then give reply
    1. Answer the question as detailed as possible from the provided context 
    2. (if not in context search on Internet), 
    3. make sure to provide all the details Properly, 
    4. use pointers and tables to make context more readable. 
    5. If information not found then search on google and then provide reply.
    6. (but then mention the reference name)
    7. If 'Summarize' word is used in input then Summarize the context.
    8. If input is: 'Hello', reply: Hey hi Suraj.\n\n
    9. Use Markdown font to make text more readable
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Updated to use gemini-2.0-flash model
    model_name = "models/gemini-2.0-flash"
    
    # Check if the model is available
    if model_name not in available_models and model_name[7:] not in [m[7:] if m.startswith("models/") else m for m in available_models]:
        # Fallback to another available model
        fallback_models = ["models/gemini-1.5-pro", "models/gemini-pro", "gemini-pro"]
        for name in fallback_models:
            if name in available_models or (name.startswith("models/") and name[7:] in [m[7:] if m.startswith("models/") else m for m in available_models]):
                model_name = name
                break
    
    print(f"Using model: {model_name}")
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use the modern LangChain approach - create a document chain
    document_chain = create_stuff_documents_chain(model, prompt)

    return document_chain

def user_input(user_question):
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever()
        docs = vector_store.similarity_search(user_question)

        document_chain = asyncio.run(get_conversational_chain())
        
        # Use the modern approach to process the query
        response = document_chain.invoke({
            "context": docs,
            "question": user_question
        })

        st.session_state.output_text = response
        st.write("Reply: ", st.session_state.output_text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def main():
    st.write("<h1><center>One-Click Conversions</center></h1>", unsafe_allow_html=True)
    st.write("")
    
    # Load animation if file exists
    try:
        with open('src/Robot.json', encoding='utf-8') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 100, -200)
    except Exception as e:
        st.warning(f"Animation file not found or couldn't be loaded: {str(e)}")

    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""

    if 'prompt_selected' not in st.session_state:
        st.session_state.prompt_selected = ""

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if st.button("Train & Process"):
        if pdf_docs:
            with st.spinner("🤖Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done, AI is trained")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        else:
            st.warning("Please upload PDF files first.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    enter_button = st.button('Enter')

    if enter_button or st.session_state.prompt_selected:
        if st.session_state.prompt_selected:
            user_question = st.session_state.prompt_selected
            st.session_state.prompt_selected = ""
        if st.session_state.user_question != user_question:
            st.session_state.user_question = user_question
            st.session_state.output_text = ""  # Reset output text when input changes

        if user_question:
            user_input(user_question)
        else:
            st.warning("Please enter a question.")

    if pdf_docs:
        st.session_state.pdf_docs = pdf_docs

if __name__ == "__main__":
    main()