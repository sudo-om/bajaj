from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import io
import re

# --- Configuration ---
# 🔐 Load API keys from environment variables for security
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# This is the secret key clients must provide in the Authorization header
AUTH_API_KEY = os.getenv("AUTH_API_KEY")

# Check if the API keys are available and raise an error if not.
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable not set.")
if not AUTH_API_KEY:
    raise ValueError("AUTH_API_KEY environment variable not set. This is required for client authentication.")

# Configure the generative AI model
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Authentication Setup ---
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Dependency to verify the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Q&A API",
    description="Ask questions about a document provided via URL."
)

# --- Utility Functions ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF content provided as bytes."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf_document])
        pdf_document.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF file: {e}")

def extract_text_from_url(url: str) -> str:
    """Downloads a file from a URL and extracts its text content."""
    try:
        response = requests.get(url, timeout=30) # Increased timeout for larger files
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {e}")

    ext = ''
    content_disposition = response.headers.get('content-disposition')
    if content_disposition:
        filenames = re.findall('filename="(.+)"', content_disposition)
        if filenames:
            ext = os.path.splitext(filenames[0])[1].lower().replace('.', '')

    if not ext:
        path = requests.utils.urlparse(url).path
        ext = os.path.splitext(path)[1].lower().replace('.', '')

    if ext == "pdf":
        return extract_text_from_pdf_bytes(response.content)
    elif ext == "txt":
        return response.content.decode("utf-8")
    else:
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            return extract_text_from_pdf_bytes(response.content)
        elif 'text/plain' in content_type:
            return response.content.decode("utf-8")
        
        raise HTTPException(status_code=400, detail=f"Unsupported file type from URL. Could not determine if PDF or TXT. Detected extension: '{ext}', Content-Type: '{content_type}'")


# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    questions: List[str]
    documents: str # This is now a required field

class QueryResponse(BaseModel):
    answers: List[str] # The response now only contains the answers


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(data: QueryRequest, api_key: str = Security(verify_api_key)):
    """
    Main endpoint to ask questions about a document.
    Requires Bearer token authentication.
    The 'documents' field with a URL is mandatory.
    """
    content_to_process = ""

    try:
        print(f"Processing query with URL: {data.documents}")
        content_to_process = extract_text_from_url(data.documents)

        if not content_to_process:
             raise HTTPException(
                status_code=404,
                detail="Document content is empty. Cannot process the query."
            )

        answers = []
        for q in data.questions:
            prompt = f"""
            Based *only* on the content provided below, answer the following question.
            If the answer cannot be found in the content, say "The answer is not available in the provided document."

            Question: "{q}"

            --- Document Content ---
            {content_to_process}
            """
            response = model.generate_content(prompt)
            answers.append(response.text.strip())

        # Return the response in the specified format
        return {"answers": answers}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

