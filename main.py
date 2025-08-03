from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import io
import re
from contextlib import asynccontextmanager

# --- Configuration ---
# 🔐 Load the Gemini API key from environment variables for security
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Check if the API key is available and raise an error if not.
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable not set. Please set it in your Vercel project settings.")

# Configure the generative AI model
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Global State ---
# This variable will hold the text content of the default document. It acts as a cache.
CURRENT_DOCUMENT_TEXT: Optional[str] = None
# The source for the default document, which will be loaded on demand.
DEFAULT_DOCUMENT_SOURCE = "https://drive.google.com/uc?export=download&id=1-8FgKwnbEgbJ7J7vdmkBM3qZDZOTaUTN" 

# --- FastAPI App Initialization ---
# The lifespan manager is removed as loading now happens on-demand.
app = FastAPI(
    title="Document Q&A API",
    description="Ask questions about a document provided via URL, file upload, or a default pre-loaded document."
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
        response = requests.get(url, timeout=20) # Increased timeout for serverless cold starts
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
    documents: Optional[str] = None

class QueryResponse(BaseModel):
    answers: List[str]
    source_document: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    char_count: int


# --- API Endpoints ---
@app.post("/api/v1/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file.
    This uploaded PDF will become the new context for queries that don't specify a URL.
    """
    global CURRENT_DOCUMENT_TEXT
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    print(f"Received upload: {file.filename}")
    pdf_bytes = await file.read()
    CURRENT_DOCUMENT_TEXT = extract_text_from_pdf_bytes(pdf_bytes)
    print(f"Successfully processed and updated context from '{file.filename}'.")

    return {
        "message": "PDF processed successfully. It is now the default context for queries.",
        "filename": file.filename,
        "char_count": len(CURRENT_DOCUMENT_TEXT)
    }


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(data: QueryRequest):
    """
    Main endpoint to ask questions.
    - If 'documents' URL is provided, it fetches and uses that document.
    - If 'documents' is not provided, it falls back to the default document,
      loading it on-demand if necessary.
    """
    global CURRENT_DOCUMENT_TEXT
    content_to_process = ""
    source = ""

    try:
        if data.documents:
            # Case 1: A URL is provided in the request. Use it directly.
            print(f"Processing query with URL: {data.documents}")
            source = f"URL: {data.documents}"
            content_to_process = extract_text_from_url(data.documents)
        else:
            # Case 2: No URL provided. Use the default document.
            # Check if the default document is already loaded in memory (cached).
            if not CURRENT_DOCUMENT_TEXT:
                print("Default document not in memory. Loading from source...")
                try:
                    CURRENT_DOCUMENT_TEXT = extract_text_from_url(DEFAULT_DOCUMENT_SOURCE)
                    print("Default document loaded and cached successfully.")
                except Exception as e:
                    print(f"CRITICAL: Could not load the default document on demand: {e}")
                    raise HTTPException(status_code=503, detail=f"The default document is currently unavailable: {e}")

            # At this point, the default text should be loaded.
            print("Processing query with default document.")
            source = f"Default Document: {DEFAULT_DOCUMENT_SOURCE}"
            content_to_process = CURRENT_DOCUMENT_TEXT

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

        return {"answers": answers, "source_document": source}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
