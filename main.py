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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn

# Import modules
from Source.Services.files_DocAI_processor import document_to_markdown
from Source.Services.summarizer import ContentSummarizer
from Source.Services.video_indexer_processor import VideoIndexerManager
from Source.Services.unified_indexer import index_content_files, UnifiedContentIndexer

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

class ProcessDocumentRequest(BaseModel):
    course_id: str
    section_id: str
    file_id: int
    document_name: str
    document_url: str

class ProcessVideoRequest(BaseModel):
    course_id: str
    section_id: str
    file_id: int
    video_name: str
    video_url: str
    merge_segments_duration: Optional[int] = 30

class IndexRequest(BaseModel):
    blob_paths: List[str]
    create_new_index: bool = False

class SummarizeRequest(BaseModel):
    blob_path: str

class SummarizeSectionRequest(BaseModel):
    full_blob_path: str

class SummarizeCourseRequest(BaseModel):
    full_blob_path: str

class DeleteContentRequest(BaseModel):
    source_id: str
    content_type: Optional[str] = None

class UpdateContentRequest(BaseModel):
    blob_path: str
    force_update: bool = False

class DeleteContentResponse(BaseModel):
    success: bool
    deleted_count: int
    source_id: str

class UpdateContentResponse(BaseModel):
    success: bool
    source_id: str
    old_chunks: int
    new_chunks: int

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
            "🗑️ /index/delete-content - Delete content from search index",
            "🔄 /index/update-content - Update content in search index",
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
async def process_document_file(request: ProcessDocumentRequest):
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

    **Request Body Example:**
    ```json
    {
        "course_id": "CS101",
        "section_id": "Section1",
        "file_id": 1,
        "document_name": "בדידה תרגול 02",
        "document_url": "bdida_tirgul_02.pdf"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        # Process document from blob storage with new parameters
        result_blob_path = document_to_markdown(
            request.course_id,
            request.section_id,
            request.file_id,
            request.document_name,
            request.document_url
        )

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
async def process_video_file(request: ProcessVideoRequest):
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

    **Request Body Example:**
    ```json
    {
        "course_id": "CS101",
        "section_id": "Section1",
        "file_id": 2,
        "video_name": "שיעור ראשון - חתוך",
        "video_url": "L1_091004f349688522f773afc884451c9af6da18fb_Trim.mp4",
        "merge_segments_duration": 30
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        # Process video from blob storage with new parameters
        result_blob_path = video_processor.process_video_to_md(
            request.course_id,
            request.section_id,
            request.file_id,
            request.video_name,
            request.video_url,
            request.merge_segments_duration
        )

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
async def index_content_files_endpoint(request: IndexRequest):
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

    **Request Body Example:**
    ```json
    {
        "blob_paths": [
            "CS101/Section1/Videos_md/2.md",
            "CS101/Section1/Docs_md/1.md"
        ],
        "create_new_index": false
    }
    ```

    **File Examples:**
    - **Document**: "CS101/Section1/Docs_md/bdida_tirgul_02.md"
    - **Video**: "CS101/Section1/Videos_md/L1_091004f349688522f773afc884451c9af6da18fb_Trim.md"

    **Returns:**
    - message: Result message from the indexing operation
    - create_new_index: Boolean indicating whether a new index was created
    """
    try:
        # Validate blob paths
        if not request.blob_paths:
            raise HTTPException(status_code=400, detail="Blob paths list cannot be empty")

        # Check all files are MD
        for blob_path in request.blob_paths:
            if not blob_path.lower().endswith('.md'):
                raise HTTPException(status_code=400, detail=f"Only MD files are supported: {blob_path}")

        # Use the unified_indexer to process files from blob storage
        result = index_content_files(request.blob_paths, create_new_index=request.create_new_index)

        return {
            "message": result,
            "create_new_index": request.create_new_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing files: {str(e)}")


@app.post(
    "/index/delete-content",
    response_model=DeleteContentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Content not found"},
        500: {"model": ErrorResponse, "description": "Delete operation failed"}
    },
    tags=["Indexing"]
)
async def delete_content_from_index(request: DeleteContentRequest):
    """
    🗑️ Delete Content from Search Index by Source ID

    **Function Description:**
    Removes all chunks related to a specific source (video or document) from the search index.

    **Usage Instructions:**
    1. **Provide Source ID**: The unique identifier of the content to delete
    2. **Content Type (Optional)**: Specify 'video' or 'document' to limit deletion to specific type
    3. **Safety**: All chunks belonging to the source will be permanently removed

    **What the Function Does:**
    • Searches for all chunks matching the source_id
    • Removes all matching chunks from the search index
    • Returns detailed deletion statistics

    **Request Body Example:**
    ```json
    {
        "source_id": "video_123",
        "content_type": "video"
    }
    ```

    **Use Cases:**
    - Remove content that should no longer be searchable

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - deleted_count: Number of chunks that were deleted
    - source_id: The source ID that was processed
    """
    try:
        indexer = UnifiedContentIndexer()

        # Perform deletion
        result = indexer.delete_content_by_source(
            source_id=request.source_id,
            content_type=request.content_type
        )

        if result["success"]:
            # Check if anything was actually deleted
            deleted_count = result["deleted_count"]
            if deleted_count > 0:
                return DeleteContentResponse(
                    success=True,
                    deleted_count=deleted_count,
                    source_id=request.source_id
                )
            else:
                # No content was found to delete
                raise HTTPException(
                    status_code=404,
                    detail=f"No content found for source_id: {request.source_id}"
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Delete operation failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting content: {str(e)}")


@app.post(
    "/index/update-content",
    response_model=UpdateContentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Content file not found"},
        500: {"model": ErrorResponse, "description": "Update operation failed"}
    },
    tags=["Indexing"]
)
async def update_content_in_index(request: UpdateContentRequest):
    """
    🔄 Update Content in Search Index

    **Function Description:**
    Updates existing content in the search index by removing the old version and adding the new version.

    **Usage Instructions:**
    1. **Provide Blob Path**: Path to the updated MD file in blob storage
    2. **Force Update**: Set to true to add content even if it doesn't exist in index

    **What the Function Does:**
    • Identifies the source from the blob path
    • Removes all existing chunks for that source
    • Processes the updated file and creates new chunks
    • Adds the new chunks to the search index
    • Returns statistics about the update operation

    **Request Body Example:**
    ```json
    {
        "blob_path": "CS101/Section1/Videos_md/2.md",
        "force_update": false
    }
    ```

    **Use Cases:**
    - Replace content with updated transcriptions or documents

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - source_id: The source ID that was updated
    - old_chunks: Number of chunks in the previous version
    - new_chunks: Number of chunks in the updated version
    """
    try:
        indexer = UnifiedContentIndexer()

        # Perform update
        result = indexer.update_content_file(
            blob_path=request.blob_path,
            force_update=request.force_update
        )

        if result["success"]:
            return UpdateContentResponse(
                success=True,
                source_id=result["source_id"],
                old_chunks=result.get("old_chunks", 0),
                new_chunks=result.get("new_chunks", 0)
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Update operation failed")
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating content: {str(e)}")

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

async def summarize_md_file(request: SummarizeRequest):
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

    **Request Body Example:**
    ```json
    {
        "blob_path": "CS101/Section1/Videos_md/2.md"
    }
    ```

    **Content Type Detection:**
    - Files in 'Videos_md' folders are treated as video content
    - Files in 'Docs_md' folders are treated as document content

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage (or None if failed)
    """
    try:
        # Check if file is MD
        if not request.blob_path.lower().endswith('.md'):
            return {
                "success": False,
                "blob_path": None
            }

        # Use summarizer.summarize_md_file function - it handles everything internally
        result_blob_path = summarizer.summarize_md_file(request.blob_path)

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

async def summarize_section_from_blob(request: SummarizeSectionRequest):
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

    **Request Body Example:**
    ```json
    {
        "full_blob_path": "CS101/Section1/file_summaries"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        result_blob_path = summarizer.summarize_section_from_blob(request.full_blob_path)


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
async def summarize_course_from_blob(request: SummarizeCourseRequest):
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

    **Request Body Example:**
    ```json
    {
        "full_blob_path": "CS101/section_summaries"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        result_blob_path = summarizer.summarize_course_from_blob(request.full_blob_path)

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
