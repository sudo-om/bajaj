from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import tempfile
import os
import io
import re
from contextlib import asynccontextmanager

# --- Configuration ---
# 🔐 Load the Gemini API key from environment variables for security
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Check if the API key is available and raise an error if not.
# This error will be visible in Vercel logs if the environment variable is not set.
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable not set. Please set it in your Vercel project settings.")

# Configure the generative AI model
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Global State ---
# This variable will hold the text content of the currently active PDF.
# It's initialized with the content of the default document on startup.
CURRENT_DOCUMENT_TEXT: Optional[str] = None
# This can now be a local file path OR a URL to a PDF/TXT file.
# Using the direct download link for the Google Drive file.
DEFAULT_DOCUMENT_SOURCE = "https://drive.google.com/uc?export=download&id=1-8FgKwnbEgbJ7J7vdmkBM3qZDZOTaUTN" 

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Loads the default document from a local path or a URL.
    """
    # --- Code to run on startup ---
    global CURRENT_DOCUMENT_TEXT
    print("Application startup: Loading default document...")
    try:
        # Check if the source is a URL
        if DEFAULT_DOCUMENT_SOURCE.startswith("http://") or DEFAULT_DOCUMENT_SOURCE.startswith("https://"):
            print(f"Loading default document from URL: {DEFAULT_DOCUMENT_SOURCE}")
            CURRENT_DOCUMENT_TEXT = extract_text_from_url(DEFAULT_DOCUMENT_SOURCE)
            print("Default document from URL loaded and text extracted successfully.")
        # Otherwise, treat it as a local file path
        else:
            print(f"Loading default document from local path: {DEFAULT_DOCUMENT_SOURCE}")
            if not os.path.exists(DEFAULT_DOCUMENT_SOURCE):
                print(f"WARNING: Default local document '{DEFAULT_DOCUMENT_SOURCE}' not found. The fallback mechanism will not work until a file is uploaded.")
                CURRENT_DOCUMENT_TEXT = None
            else:
                with open(DEFAULT_DOCUMENT_SOURCE, "rb") as f:
                    content_bytes = f.read()
                
                ext = DEFAULT_DOCUMENT_SOURCE.split(".")[-1].lower()
                if ext == 'pdf':
                    CURRENT_DOCUMENT_TEXT = extract_text_from_pdf_bytes(content_bytes)
                elif ext == 'txt':
                    CURRENT_DOCUMENT_TEXT = content_bytes.decode('utf-8')
                else:
                    print(f"WARNING: Unsupported local file type '{ext}'. Only PDF and TXT supported for default document.")
                    CURRENT_DOCUMENT_TEXT = None
                
                if CURRENT_DOCUMENT_TEXT:
                    print("Default local document loaded and text extracted successfully.")

    except Exception as e:
        print(f"ERROR: Could not load or process the default document on startup: {e}")
        CURRENT_DOCUMENT_TEXT = None
    
    yield # The application runs while the yield is active

    # --- Code to run on shutdown (if any) ---
    print("Application shutdown.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Q&A API",
    description="Ask questions about a document provided via URL, file upload, or a default pre-loaded document.",
    lifespan=lifespan # Register the lifespan handler
)


# --- Utility Functions ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF content provided as bytes."""
    try:
        # Open the PDF from a memory stream
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf_document])
        pdf_document.close()
        return text
    except Exception as e:
        # This will catch errors if the file is not a valid PDF
        raise HTTPException(status_code=400, detail=f"Failed to process PDF file: {e}")

def extract_text_from_url(url: str) -> str:
    """Downloads a file from a URL and extracts its text content."""
    try:
        response = requests.get(url, timeout=15) # Increased timeout for larger files
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {e}")

    # --- Robust File Type Detection ---
    ext = ''
    # 1. Try to get extension from Content-Disposition header (most reliable for cloud storage links)
    content_disposition = response.headers.get('content-disposition')
    if content_disposition:
        filenames = re.findall('filename="(.+)"', content_disposition)
        if filenames:
            ext = os.path.splitext(filenames[0])[1].lower().replace('.', '')

    # 2. If not found, try to get it from the URL path
    if not ext:
        path = requests.utils.urlparse(url).path
        ext = os.path.splitext(path)[1].lower().replace('.', '')

    # --- Process based on extension ---
    if ext == "pdf":
        return extract_text_from_pdf_bytes(response.content)
    elif ext == "txt":
        return response.content.decode("utf-8")
    else:
        # 3. As a last resort, check the Content-Type header
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            return extract_text_from_pdf_bytes(response.content)
        elif 'text/plain' in content_type:
            return response.content.decode("utf-8")
        
        raise HTTPException(status_code=400, detail=f"Unsupported file type from URL. Could not determine if PDF or TXT. Detected extension: '{ext}', Content-Type: '{content_type}'")


# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    questions: List[str]
    # The document URL is now optional.
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
    # Update the global variable with the text from the new PDF
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
    - If 'documents' is not provided, it falls back to the currently active document
      (either the one uploaded or the default one loaded at startup).
    """
    content_to_process = ""
    source = ""

    try:
        if data.documents:
            # Case 1: A URL is provided in the request.
            print(f"Processing query with URL: {data.documents}")
            source = f"URL: {data.documents}"
            content_to_process = extract_text_from_url(data.documents)
        elif CURRENT_DOCUMENT_TEXT:
            # Case 2: No URL is provided, use the globally stored text.
            print("Processing query with pre-loaded/uploaded document.")
            source = "Pre-loaded or Uploaded Document"
            content_to_process = CURRENT_DOCUMENT_TEXT
        else:
            # Case 3: No URL and no document has been loaded.
            raise HTTPException(
                status_code=404,
                detail="No document context available. Please provide a document URL or upload a PDF first."
            )

        answers = []
        for q in data.questions:
            # Construct a clear prompt for the model
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
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
