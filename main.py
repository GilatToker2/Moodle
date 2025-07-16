"""
🎓 Academic Content Processing API

📖 API Documentation: http://localhost:8080/docs

🔑 Required old_config.py settings:
- STORAGE_CONNECTION_STRING
- CONTAINER_NAME ("course")
- AZURE_OPENAI_API_KEY
- VIDEO_INDEXER_ACCOUNT_ID
- SEARCH_SERVICE_NAME
"""

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn

# Import modules
from Source.Services.files_DocAI_processor import document_to_markdown
from Source.Services.summarizer import ContentSummarizer
from Source.Services.video_indexer_processor import VideoIndexerManager
from Source.Services.unified_indexer import index_content_files

# Initialize FastAPI app
app = FastAPI(title="Academic Content API", version="1.0.0")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
summarizer = ContentSummarizer()
video_processor = VideoIndexerManager()

# ================================
# 📋 RESPONSE MODELS
# ================================

class ErrorResponse(BaseModel):
    detail: str

class ProcessDocumentResponse(BaseModel):
    success: bool
    blob_path: Optional[str] = None

class ProcessVideoResponse(BaseModel):
    success: bool
    blob_path: Optional[str] = None

class IndexResponse(BaseModel):
    message: str
    create_new_index: bool

class SummarizeResponse(BaseModel):
    success: bool
    blob_path: Optional[str] = None

# ================================
# 🏠 ROOT & HEALTH ENDPOINTS
# ================================

@app.get("/", tags=["System"])
async def root():
    """דף בית - מידע כללי על המערכת"""
    return {
        "message": "🎓 מערכת עיבוד תוכן אקדמי",
        "version": "1.0.0",
        "status": "פעיל",
        "functions": [
            "📄 /process/document - Convert documents to Markdown",
            "🎥 /process/video - Process videos with transcription",
            "🗂️ /index/content-files - Index files for search",
            "📝 /summarize/md - Create summary from Markdown",
            "📚 /summarize/section - Create section summary",
            "🎓 /summarize/course - Create course summary"
        ],
        "docs_url": "/docs"
    }

# ================================
# 📄 DOCUMENT PROCESSING ENDPOINTS
# ================================

@app.post(
    "/process/document",
    response_model=ProcessDocumentResponse,
    responses={500: {"model": ErrorResponse, "description": "Document processing failed"}},
    tags=["Document Processing"]
)
async def process_document_file(
    course_id: str = Form(..., description="מזהה הקורס (למשל: CS101)"),
    section_id: str = Form(..., description="מזהה הסקציה (למשל: Section1)"),
    file_id: int = Form(..., description="מזהה הקובץ (למשל: 1)"),
    document_name: str = Form(..., description="שם המסמך (למשל: בדידה תרגול 02)"),
    document_url: str = Form(..., description="נתיב הקובץ ב-blob storage (למשל: bdida_tirgul_02.pdf)")
):
    """
    📄 Process Document to Markdown Format

    **Function Description:**
    Converts documents from blob storage to Markdown format using Azure Document AI.

    **Supported Formats:**
    PDF, DOCX, PPTX, PNG, JPG

    **What to Expect:**
    • Document is downloaded from blob storage and processed in memory
    • Uses Azure Document AI for intelligent text extraction
    • Saves markdown to blob storage with new structure: CourseID/SectionID/Docs_md/FileID.md
    • Document name is included in the transcription
    • Returns structured Markdown content

    **Parameters:**
    - course_id: Course identifier (e.g., "CS101")
    - section_id: Section identifier (e.g., "Section1")
    - file_id: File identifier (e.g., 1)
    - document_name: Document name (e.g., "בדידה תרגול 02")
    - document_url: Path to document in blob storage (e.g., "Raw-data/bdida_tirgul_02.pdf")

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        # Process document from blob storage with new parameters
        result_blob_path = document_to_markdown(course_id, section_id, file_id, document_name, document_url)

        if result_blob_path:
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        return {
            "success": False,
            "blob_path": None
        }

# ================================
# 🎥 VIDEO PROCESSING ENDPOINTS
# ================================

@app.post(
    "/process/video",
    response_model=ProcessVideoResponse,
    responses={500: {"model": ErrorResponse, "description": "Video processing failed"}},
    tags=["Video Processing"]
)
async def process_video_file(
    course_id: str = Form(..., description="מזהה הקורס (למשל: CS101)"),
    section_id: str = Form(..., description="מזהה הסקציה (למשל: Section1)"),
    file_id: int = Form(..., description="מזהה הקובץ (למשל: 1)"),
    video_name: str = Form(..., description="שם הוידאו (למשל: שיעור 1 - מבוא למתמטיקה דיסקרטית)"),
    video_url: str = Form(..., description="נתיב הוידאו ב-blob storage (למשל: L1_091004f349688522f773afc884451c9af6da18fb_Trim.mp4)"),
    merge_segments_duration: Optional[int] = Form(30, description="Duration in seconds for merging transcript segments")
):
    """
    🎥 Process Video File with Azure Video Indexer

    **Function Description:**
    Processes video from blob storage using Azure Video Indexer for transcription and analysis, then converts to Markdown format.

    **What to Expect:**
    • Video is processed from blob storage using Video Indexer
    • Automatic Hebrew transcription with timestamps
    • AI-powered extraction of keywords and topics
    • Subject type classification ("הומני" or "מתמטי")
    • Short automatic lesson summary generation
    • Full transcript with precise timestamps
    • Saves markdown to blob storage with new structure: CourseID/SectionID/Videos_md/FileID.md
    • Video name is included in the transcription

    **Parameters:**
    - course_id: Course identifier (e.g., "CS101")
    - section_id: Section identifier (e.g., "Section1")
    - file_id: File identifier (e.g., 2)
    - video_name: Video name (e.g., "שיעור ראשון - חתוך")
    - video_url: Path to video in blob storage (e.g., "L1_091004f349688522f773afc884451c9af6da18fb_Trim.mp4")
    - merge_segments_duration: Duration in seconds for merging transcript segments (default: 30)

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        # Process video from blob storage with new parameters
        result_blob_path = video_processor.process_video_to_md(course_id, section_id, file_id, video_name, video_url, merge_segments_duration)

        if result_blob_path:
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        return {
            "success": False,
            "blob_path": None
        }

# ================================
# 🗂️ INDEXING ENDPOINTS
# ================================

@app.post(
    "/index/content-files",
    response_model=IndexResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or content type"},
        500: {"model": ErrorResponse, "description": "Indexing failed"}
    },
    tags=["Indexing"]
)
async def index_content_files_endpoint(
        blob_paths: List[str] = Form(..., description="List of blob paths to MD files (e.g., ['CS101/Section1/Videos_md/2.md','CS101/Section1/Docs_md/1.md'])"),
        create_new_index: bool = Form(False, description="Create new index? (default: False)")
):
    """
    🗂️ Index MD Files from Blob Storage to Search Index

    **Function Description:**
    Indexes multiple Markdown files from blob storage into Azure Cognitive Search for searchability.

    **Usage Instructions:**
    1. **Provide Blob Paths**: List of blob paths to .md files you want to add to the search index
    2. **Content Type Detection**: Content type is automatically detected from path structure
       - Files in 'Videos_md' folders are treated as video content
       - Files in 'Docs_md' folders are treated as document content
    3. **New Index**: Choose whether to create a new index or add to existing

    **What the Function Does:**
    • Downloads and processes MD files from blob storage
    • Automatically detects content type from path structure
    • Splits content into small chunks for search
    • Generates embeddings for each chunk
    • Adds to Azure Search index

    **File Examples:**
    - **Document**: "CS101/Section1/Docs_md/1.md"
    - **Video**: "CS101/Section1/Videos_md/2.md"

    **Returns:**
    - message: Result message from the indexing operation
    - create_new_index: Boolean indicating whether a new index was created
    """
    try:
        # Validate blob paths
        if not blob_paths or not isinstance(blob_paths, list):
            raise HTTPException(status_code=400, detail="Blob paths must be a non-empty list")

        # Check all files are MD
        for blob_path in blob_paths:
            if not blob_path.lower().endswith('.md'):
                raise HTTPException(status_code=400, detail=f"Only MD files are supported: {blob_path}")

        # Use the unified_indexer to process files from blob storage
        result = index_content_files(blob_paths, create_new_index=create_new_index)

        return {
            "message": result,
            "create_new_index": create_new_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing files: {str(e)}")



# ================================
# 📝 SUMMARIZATION ENDPOINTS
# ================================

@app.post(
    "/summarize/md",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Summarization failed"}
    },
    tags=["Summarization"]
)
async def summarize_md_file(
        blob_path: str = Form(..., description="Path to MD file in blob storage (e.g., CS101/Section1/Videos_md/2.md)")
):
    """
    📝 Create Summary from Markdown File in Blob Storage

    **Function Description:**
    Generates an LLM-based summary from a Markdown file stored in blob storage.

    **What to Expect:**
    • Downloads MD file from blob storage
    • Automatically detects content type from path structure
    • Processes the entire markdown content
    • Returns a comprehensive summary based on content type
    • Uses Azure OpenAI for intelligent summarization
    • Saves summary back to blob storage

    **Parameters:**
    - blob_path: Path to MD file in blob storage (e.g., "CS101/Section1/Videos_md/L1_091004f349688522f773afc884451c9af6da18fb_Trim.md")

    **Content Type Detection:**
    - Files in 'Videos_md' folders are treated as video content
    - Files in 'Docs_md' folders are treated as document content

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage (or None if failed)
    """
    try:
        # Check if file is MD
        if not blob_path.lower().endswith('.md'):
            return {
                "success": False,
                "blob_path": None
            }

        # Use summarizer.summarize_md_file function - it handles everything internally
        result_blob_path = summarizer.summarize_md_file(blob_path)

        if result_blob_path:
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        return {
            "success": False,
            "blob_path": None
        }

@app.post(
    "/summarize/section",
    response_model=SummarizeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No content found"},
        500: {"model": ErrorResponse, "description": "Section summarization failed"}
    },
    tags=["Summarization"]
)

async def summarize_section_from_blob(
        full_blob_path: str = Form(..., description="Full blob path to section summaries (e.g., CS101/Section1/file_summaries)")
):
    """
    📚 Create Section Summary from Azure Storage

    **Function Description:**
    Scans all summary files in the specified blob path and creates a unified section summary.

    **What to Expect:**
    • Connects to Azure Storage
    • Scans the specified blob path for summary files
    • Processes markdown summary files (.md) from the path
    • Creates a comprehensive section-level summary
    • Uses Azure OpenAI for intelligent content analysis
    • Saves section summary back to blob storage

    **Parameters:**
    - full_blob_path: Full blob path to section summaries (e.g., "CS101/Section1/file_summaries")

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        result_blob_path = summarizer.summarize_section_from_blob(full_blob_path)

        if result_blob_path:
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        return {
            "success": False,
            "blob_path": None
        }


@app.post(
    "/summarize/course",
    response_model=SummarizeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No content found"},
        500: {"model": ErrorResponse, "description": "Course summarization failed"}
    },
    tags=["Summarization"]
)
async def summarize_course_from_blob(
    full_blob_path: str = Form(..., description="Full blob path to sections summary folder (e.g., CS101/section_summaries)")
):
    """
    🎓 Create Complete Course Summary from Azure Storage

    **Function Description:**
    Analyzes all section summary files in the specified blob path and generates a comprehensive course-level summary.

    **What to Expect:**
    • Connects to Azure Storage
    • Scans the specified blob path for section summary files
    • Processes all section summary markdown files (.md)
    • Creates a high-level course overview and summary
    • Uses Azure OpenAI for intelligent course-level analysis
    • Saves course summary back to blob storage

    **Parameters:**
    - full_blob_path: Full blob path to sections summary folder (e.g., "CS101/section_summaries")

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        result_blob_path = summarizer.summarize_course_from_blob(full_blob_path)

        if result_blob_path:
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        return {
            "success": False,
            "blob_path": None
        }

# ================================
# 🚀 הרצת השרת
# ================================

if __name__ == "__main__":

    print("🚀 Starting FastAPI server...")
    print("📖 API documentation available at: http://localhost:8080/docs")
    print("🏠 Home page: http://localhost:8080/")
    print("⏹️ Stop server: Ctrl+C")

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8080,
        log_level="info",
        reload=True
    )
