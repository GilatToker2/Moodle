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
from Source.Services.Files_DocAI_processor import document_to_markdown
from Source.Services.Summarizer import ContentSummarizer
from Source.Services.Video_indexer_processor import VideoIndexerManager
from Source.Services.Unified_indexer import index_content_files

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
    message: str
    original_filename: str
    markdown_length: int
    markdown_content: str

class ProcessVideoResponse(BaseModel):
    message: str
    original_filename: str
    video_id: str
    video_name: str
    duration: str
    transcript_segments: int
    keywords: List[str]
    topics: List[str]
    summary_available: bool
    markdown_length: int
    markdown_content: str
    structured_data: Dict[str, Any]

class IndexResponse(BaseModel):
    message: str
    filename: str
    content_type: str
    create_new_index: bool
    success: bool

class SummarizeResponse(BaseModel):
    message: str
    original_filename: str
    content_type: str
    summary_length: int
    summary: str

class SummarizeStorageResponse(BaseModel):
    container_name: str
    summary_length: int
    summary: str

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
    blob_file_path: str = Form(..., description="נתיב הקובץ ב-blob storage (למשל: Section1/Raw-data/Docs/filename.pdf)")
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
    • Saves markdown to blob storage with folder structure preservation
    • Returns structured Markdown content

    **Parameters:**
    - blob_file_path: Path to document in blob storage (e.g., "Section1/Raw-data/Docs/filename.pdf")

    **Returns:**
    - Processing status message
    - Original filename
    - Markdown content length
    - Complete Markdown content
    """
    try:
        # Process document from blob storage - function handles everything internally
        result_blob_path = document_to_markdown(blob_file_path)

        if not result_blob_path:
            raise HTTPException(status_code=500, detail="שגיאה בעיבוד המסמך")

        filename = os.path.basename(blob_file_path)

        return {
            "message": f"מסמך עובד בהצלחה ונשמר ב-blob: {result_blob_path}",
            "original_filename": filename,
            "markdown_length": 0,  # We don't download content just to get length
            "markdown_content": f"קובץ עובד בהצלחה ונשמר ב-blob: {result_blob_path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"שגיאה בעיבוד מסמך: {str(e)}")

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
    blob_video_path: str = Form(..., description="Path to video in blob storage (e.g., Section1/Raw-data/Videos/filename.mp4)"),
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
    • Saves markdown to blob storage with folder structure preservation

    **Parameters:**
    - blob_video_path: Path to video in blob storage (e.g., "Section1/Raw-data/Videos/filename.mp4")
    - merge_segments_duration: Duration in seconds for merging transcript segments (default: 30)

    **Returns:**
    - Video ID from Video Indexer
    - Video name and duration
    - Number of transcript segments
    - Extracted keywords and topics
    - Complete structured Markdown content
    - Structured video data with all metadata
    """
    try:
        # Process video from blob storage - function handles everything internally
        result_blob_path = video_processor.process_video_to_md(blob_video_path, merge_segments_duration)

        if not result_blob_path:
            raise HTTPException(status_code=500, detail="Error processing video")

        filename = os.path.basename(blob_video_path)

        return {
            "message": f"Video processed successfully and saved to blob: {result_blob_path}",
            "original_filename": filename,
            "video_id": "processed",
            "video_name": filename,
            "duration": "unknown",
            "transcript_segments": 0,
            "keywords": [],
            "topics": [],
            "summary_available": True,
            "markdown_length": 0,
            "markdown_content": f"Video processed successfully and saved to blob: {result_blob_path}",
            "structured_data": {"result_path": result_blob_path}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

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
        blob_paths: List[str] = Form(..., description="List of blob paths to MD files (e.g., ['Section1/Processed-data/Videos_md/file.md'])"),
        create_new_index: bool = Form(False, description="Create new index? (default: False)")
):
    """
    🗂️ Index MD Files from Blob Storage to Search Index

    **Function Description:**
    Indexes multiple Markdown files from blob storage into Azure Cognitive Search for searchability.

    **Usage Instructions:**
    1. **Provide Blob Paths**: List of blob paths to .md files you want to add to the search index
    2. **Content Type Detection**: Content type is automatically detected from path structure
       - Files in 'videos_md' folders are treated as video content
       - Files in 'docs_md' folders are treated as document content
    3. **New Index**: Choose whether to create a new index or add to existing

    **What the Function Does:**
    • Downloads and processes MD files from blob storage
    • Automatically detects content type from path structure
    • Splits content into small chunks for search
    • Generates embeddings for each chunk
    • Adds to Azure Search index

    **File Examples:**
    - **Document**: "Section1/Processed-data/Docs_md/bdida_tirgul_02.md"
    - **Video**: "Section1/Processed-data/Videos_md/L1_091004f349688522f773afc884451c9af6da18fb_Trim.md"

    **Returns:**
    - Success/failure message
    - Processing details
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
            "filename": f"{len(blob_paths)} files",
            "content_type": "auto-detected",
            "create_new_index": create_new_index,
            "success": "successfully" in result.lower()
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
        blob_path: str = Form(..., description="Path to MD file in blob storage (e.g., Section1/Processed-data/Videos_md/file.md)")
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
    - blob_path: Path to MD file in blob storage (e.g., "Section1/Processed-data/Videos_md/file.md")

    **Content Type Detection:**
    - Files in 'videos_md' folders are treated as video content
    - Files in 'docs_md' folders are treated as document content

    **Returns:**
    - Original filename
    - Content type used
    - Summary length (characters)
    - Generated summary text
    """
    try:
        # Check if file is MD
        if not blob_path.lower().endswith('.md'):
            raise HTTPException(status_code=400, detail="Only MD files are supported")

        # Use summarizer.summarize_md_file function - it handles everything internally
        result_blob_path = summarizer.summarize_md_file(blob_path)

        if not result_blob_path:
            raise HTTPException(status_code=500, detail="Error creating summary")

        filename = os.path.basename(blob_path)

        return {
            "message": f"Summary created successfully and saved to blob: {result_blob_path}",
            "original_filename": filename,
            "content_type": "auto-detected",
            "summary_length": 0,  # We don't download content just to get length
            "summary": f"Summary created successfully and saved to blob: {result_blob_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating summary: {str(e)}")


@app.post(
    "/summarize/section",
    response_model=SummarizeStorageResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No content found"},
        500: {"model": ErrorResponse, "description": "Section summarization failed"}
    },
    tags=["Summarization"]
)
async def summarize_section_from_blob(
        full_blob_path: str = Form(..., description="Full blob path to section summaries (e.g., course1/Summaries/Section1)")
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
    - full_blob_path: Full blob path to section summaries (e.g., "course1/Summaries/Section1")

    **Returns:**
    - Container name used
    - Summary length (characters)
    - Generated section summary text
    """
    try:
        result_blob_path = summarizer.summarize_section_from_blob(full_blob_path)

        if not result_blob_path:
            raise HTTPException(status_code=404, detail="No content found or error creating section summary")

        # Extract container name from path
        container_name = full_blob_path.split('/')[0] if '/' in full_blob_path else full_blob_path

        return {
            "container_name": container_name,
            "summary_length": 0,  # We don't download content just to get length
            "summary": f"Section summary created successfully and saved to blob: {result_blob_path}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating section summary: {str(e)}")


@app.post(
    "/summarize/course",
    response_model=SummarizeStorageResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No content found"},
        500: {"model": ErrorResponse, "description": "Course summarization failed"}
    },
    tags=["Summarization"]
)
async def summarize_course_from_blob(
    full_blob_path: str = Form(..., description="Full blob path to sections summary folder (e.g., course1/Summaries/Sections_summary)")
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
    - full_blob_path: Full blob path to sections summary folder (e.g., "course1/Summaries/Sections_summary")

    **Returns:**
    - Container name used
    - Summary length (characters)
    - Generated course summary text
    """
    try:
        result_blob_path = summarizer.summarize_course_from_blob(full_blob_path)

        if not result_blob_path:
            raise HTTPException(status_code=404, detail="No content found or error creating course summary")

        # Extract container name from path
        container_name = full_blob_path.split('/')[0] if '/' in full_blob_path else full_blob_path

        return {
            "container_name": container_name,
            "summary_length": 0,  # We don't download content just to get length
            "summary": f"Course summary created successfully and saved to blob: {result_blob_path}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating course summary: {str(e)}")



# ================================
# 🚀 הרצת השרת
# ================================

if __name__ == "__main__":

    print("🚀 Starting FastAPI server...")
    print("📖 API documentation available at: http://localhost:8080/docs")
    print("🏠 Home page: http://localhost:8080/")
    print("⏹️ Stop server: Ctrl+C")

    uvicorn.run(
        "Main:app",
        host="localhost",
        port=8080,
        log_level="info",
        reload=True
    )
