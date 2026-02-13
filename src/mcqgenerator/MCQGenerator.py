from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Access API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# List available models to pick the best one
models = genai.list_models()
available_models = [model.name for model in models]
print("Available models for MCQ generation:", available_models)

# Create a quiz generation prompt template
quiz_generation_template = """
You are an expert quiz creator specializing in creating multiple-choice questions (MCQs) for {subject} students at {tone} level.

Based on the following text, create {number} multiple-choice questions that test the understanding of key concepts.

TEXT: {text}

For each question:
1. Frame a clear question
2. Provide 4 options labeled as A, B, C, and D with only one correct answer
3. Indicate the correct answer

Format your response as a valid JSON object matching this exact structure:
{response_json}

IMPORTANT: Your response should ONLY contain the valid JSON object and nothing else - no introduction, no explanations. Just the JSON.
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=quiz_generation_template
)

# Initialize the LLM
def get_llm():
    # Try using gemini-2.0-flash first
    preferred_models = ["models/gemini-2.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
    
    model_to_use = None
    for model in preferred_models:
        if model in available_models or (model.startswith("models/") and model[7:] in [m[7:] if m.startswith("models/") else m for m in available_models]):
            model_to_use = model
            break
    
    # Fallback to any available model
    if not model_to_use and available_models:
        model_to_use = available_models[0]
    
    print(f"Using model for MCQ generation: {model_to_use}")
    llm = ChatGoogleGenerativeAI(model=model_to_use, temperature=0.3)
    return llm

# A wrapper function to ensure we get a usable response
def generate_mcq_with_fallback(input_data):
    llm = get_llm()
    response = llm.invoke(quiz_generation_prompt.format(**input_data))
    
    # Return the content directly for easier handling
    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)

# Create a chain-like interface that mimics the previous chain but is more robust
class MCQChain:
    def invoke(self, input_data):
        return generate_mcq_with_fallback(input_data)

# Create the chain
generate_evaluate_chain = MCQChain()

