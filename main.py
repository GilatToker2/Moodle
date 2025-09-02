"""
Academic Content Processing API

API Documentation: http://localhost:8080/docs

Required config.py settings:
- STORAGE_CONNECTION_STRING
- CONTAINER_NAME ("course")
- AZURE_OPENAI_API_KEY
- VIDEO_INDEXER_ACCOUNT_ID
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
from Config.logging_config import setup_logging
from Config.config import (
    AZURE_FORM_RECOGNIZER_KEY,
    AZURE_FORM_RECOGNIZER_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION
)
# Import modules
from Source.Services.files_DocAI_processor import document_to_markdown
from Source.Services.summarizer import ContentSummarizer
from Source.Services.video_indexer_processor import VideoIndexerManager
from Source.Services.unified_indexer import index_content_files, UnifiedContentIndexer
from Source.Services.subject_detector import detect_subject_from_course
from Source.Services.blob_manager import BlobManager
from Source.Services.syllabus_generator import SyllabusGenerator
from Source.Services.prompt_loader import initialize_prompt_loader

# Initialize logger
logger = setup_logging()


# Convenience function for backward compatibility
def debug_log(message):
    """Write debug message using proper logging"""
    logger.debug(message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application
    Manages startup and shutdown of Azure services and connections
    """
    # STARTUP - Initialize services and connections
    logger.info("App is starting - Initializing services...")

    try:
        # Initialize Azure Document Intelligence Client
        logger.info("Initializing Azure Document Intelligence Client...")
        di_client = DocumentIntelligenceClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )
        app.state.di_client = di_client
        logger.info("Document Intelligence Client initialized successfully")

        # Initialize Azure Blob Storage Managers
        logger.info("Initializing Azure Blob Storage Managers...")
        blob_manager = BlobManager()  # Default container (processeddata)
        blob_manager_raw = BlobManager(container_name="raw-data")
        app.state.blob_manager = blob_manager
        app.state.blob_manager_raw = blob_manager_raw
        logger.info("Blob Storage Managers initialized successfully")

        # Initialize and preload all prompts at startup FIRST
        logger.info("Initializing and preloading all prompts...")
        prompt_loader = initialize_prompt_loader()
        app.state.prompt_loader = prompt_loader
        logger.info("All prompts preloaded successfully")

        # Initialize shared OpenAI client
        logger.info("Initializing shared OpenAI client...")
        shared_openai_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        app.state.shared_openai_client = shared_openai_client
        logger.info("Shared OpenAI client initialized successfully")

        # Initialize Content Summarizer (now with shared OpenAI client)
        logger.info("Initializing Content Summarizer...")
        summarizer = ContentSummarizer(
            prompt_loader=prompt_loader,
            blob_manager=blob_manager,
            openai_client=shared_openai_client
        )
        app.state.summarizer = summarizer
        logger.info("Content Summarizer initialized successfully")

        # Initialize Subject Detector (also needs shared OpenAI client)
        logger.info("Initializing Subject Detector...")
        from Source.Services.subject_detector import SubjectDetector
        subject_detector = SubjectDetector(
            prompt_loader=prompt_loader,
            blob_manager=blob_manager,
            openai_client=shared_openai_client
        )
        app.state.subject_detector = subject_detector
        logger.info("Subject Detector initialized successfully")

        # Initialize Video Indexer Manager
        logger.info("Initializing Video Indexer Manager...")
        video_processor = VideoIndexerManager()
        app.state.video_processor = video_processor
        logger.info("Video Indexer Manager initialized successfully")

        # Initialize Unified Content Indexer (with shared OpenAI client)
        logger.info("Initializing Unified Content Indexer...")
        content_indexer = UnifiedContentIndexer(openai_client=shared_openai_client)
        app.state.content_indexer = content_indexer
        logger.info("Unified Content Indexer initialized successfully")

        # Initialize Syllabus Generator (with shared OpenAI client)
        logger.info("Initializing Syllabus Generator...")
        syllabus_generator = SyllabusGenerator(
            prompt_loader=prompt_loader,
            blob_manager=blob_manager,
            openai_client=shared_openai_client
        )
        app.state.syllabus_generator = syllabus_generator
        logger.info("Syllabus Generator initialized successfully")
        logger.info("All services initialized successfully - Application ready!")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    # Application is running
    yield

    # SHUTDOWN - Clean up resources and close connections
    logger.info("App is shutting down - Cleaning up resources...")

    try:
        # First, close the shared OpenAI client (this is the root cause of connection issues)
        if hasattr(app.state, "shared_openai_client"):
            shared_openai_client = app.state.shared_openai_client
            logger.info("Closing shared OpenAI client...")
            try:
                await shared_openai_client.close()
                logger.info("Shared OpenAI client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing shared OpenAI client: {e}")

        # Close Azure Document Intelligence Client
        if hasattr(app.state, "di_client"):
            di_client = app.state.di_client
            logger.info("Closing Document Intelligence Client...")
            try:
                await di_client.close()
                logger.info("Document Intelligence Client closed")
            except Exception as e:
                logger.warning(f"Error closing Document Intelligence Client: {e}")

        # Close Azure Blob Storage connections
        if hasattr(app.state, "blob_manager"):
            blob_manager = app.state.blob_manager
            logger.info("Closing Azure Blob Storage connections (processed data)...")
            try:
                await blob_manager.close()
                logger.info("Blob Storage connections (processed data) closed")
            except Exception as e:
                logger.warning(f"Error closing Blob Storage connections (processed data): {e}")

        # Close Azure Blob Storage connections for raw data
        if hasattr(app.state, "blob_manager_raw"):
            blob_manager_raw = app.state.blob_manager_raw
            logger.info("Closing Azure Blob Storage connections (raw data)...")
            try:
                await blob_manager_raw.close()
                logger.info("Blob Storage connections (raw data) closed")
            except Exception as e:
                logger.warning(f"Error closing Blob Storage connections (raw data): {e}")

        # Clean up services (but don't close shared OpenAI client - already closed above)
        if hasattr(app.state, "content_indexer"):
            content_indexer = app.state.content_indexer
            if hasattr(content_indexer, 'close'):
                logger.info("Cleaning up Content Indexer...")
                try:
                    await content_indexer.close()
                    logger.info("Content Indexer cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up Content Indexer: {e}")

        if hasattr(app.state, "summarizer"):
            summarizer = app.state.summarizer
            if hasattr(summarizer, 'close'):
                logger.info("Cleaning up Summarizer...")
                try:
                    await summarizer.close()
                    logger.info("Summarizer cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up Summarizer: {e}")

        # Close Video Indexer Manager connections
        if hasattr(app.state, "video_processor"):
            video_processor = app.state.video_processor
            if hasattr(video_processor, 'close') and callable(video_processor.close):
                logger.info("Closing Video Indexer Manager...")
                try:
                    await video_processor.close()
                    logger.info("Video Indexer Manager closed")
                except Exception as e:
                    logger.warning(f"Error closing Video Indexer Manager: {e}")

        logger.info("All resources cleaned up successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("Application shutdown completed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Academic Content API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get services from app.state (initialized in lifespan)
def get_summarizer():
    return getattr(app.state, "summarizer", None) or ContentSummarizer()


def get_video_processor():
    return getattr(app.state, "video_processor", None) or VideoIndexerManager()


def get_blob_manager():
    return getattr(app.state, "blob_manager", None) or BlobManager()


def get_blob_manager_raw():
    return getattr(app.state, "blob_manager_raw", None) or BlobManager(container_name="raw-data")


def get_content_indexer():
    return getattr(app.state, "content_indexer", None) or UnifiedContentIndexer()


def get_di_client():
    """Get the shared Document Intelligence client"""
    return getattr(app.state, "di_client", None)


def get_syllabus_generator():
    """Get the shared Syllabus Generator"""
    return getattr(app.state, "syllabus_generator", None) or SyllabusGenerator()


def get_prompt_service():
    """Get the shared Prompt Loader"""
    from Source.Services.prompt_loader import get_prompt_loader as fallback_get_prompt_loader
    return getattr(app.state, "prompt_loader", None) or fallback_get_prompt_loader()


def get_subject_detector():
    """Get the shared Subject Detector"""
    from Source.Services.subject_detector import SubjectDetector
    return getattr(app.state, "subject_detector", None) or SubjectDetector()


# ================================
# RESPONSE MODELS
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


class ProcessVideoFromIdRequest(BaseModel):
    course_id: str
    section_id: str
    file_id: int
    video_name: str
    video_id: str


class IndexRequest(BaseModel):
    blob_paths: List[str]
    create_new_index: bool = False


class SummarizeRequest(BaseModel):
    blob_path: str
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None


class SummarizeFilesRequest(BaseModel):
    blob_paths: List[str]
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None


class SummarizeFilesResponse(BaseModel):
    success: bool
    results: Dict[str, Optional[str]]
    total_processed: int
    successful: int
    failed: int


class SummarizeSectionRequest(BaseModel):
    full_blob_path: str
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None
    previous_summary_path: Optional[str] = None


class SummarizeCourseRequest(BaseModel):
    full_blob_path: str
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None


class DeleteContentRequest(BaseModel):
    source_id: str
    content_type: Optional[str] = None


class DeleteContentResponse(BaseModel):
    success: bool
    deleted_count: int
    source_id: str


class DetectSubjectRequest(BaseModel):
    course_path: str


class DetectSubjectResponse(BaseModel):
    success: bool
    subject_name: str
    subject_type: str


class CreateSyllabusRequest(BaseModel):
    full_blob_path: str
    subject_name: Optional[str] = None
    subject_type: Optional[str] = None


# ================================
# ROOT & HEALTH ENDPOINTS
# ================================

@app.get("/", tags=["System"])
async def root():
    """Home page - General system information"""
    return {
        "message": "Academic Content Processing System",
        "version": "1.0.0",
        "status": "Active",
        "functions": [
            "/process/document - Convert documents to Markdown",
            "/process/video - Process videos with transcription",
            "/process/video_from_id - Process videos from existing Video Indexer ID",
            "/insert_to_index - Insert files to search index",
            "/delete_from_index - Delete content from search index",
            "/summarize/md - Create summary from Markdown",
            "/summarize/md_files - Create summaries from multiple Markdown files (batch)",
            "/summarize/section - Create section summary",
            "/summarize/course - Create course summary",
            "/detect/subject - Detect subject type from course",
            "/create/syllabus - Create academic syllabus from course summary"
        ],
        "docs_url": "/docs"
    }


# ================================
# DOCUMENT PROCESSING ENDPOINTS
# ================================

@app.post(
    "/process/document",
    response_model=ProcessDocumentResponse,
    responses={500: {"model": ErrorResponse, "description": "Document processing failed"}},
    tags=["Document Processing"]
)
async def process_document_file(request: ProcessDocumentRequest):
    """
    Process Document to Markdown Format

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
        "document_name": "Discrete Math Exercise 02",
        "document_url": "bdida_tirgul_02.pdf"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        logger.info(f"Starting document processing: {request.document_name}")
        logger.info(f"CourseID: {request.course_id}, SectionID: {request.section_id}, FileID: {request.file_id}")
        logger.info(f"DocumentURL: {request.document_url}")

        # Get shared Document Intelligence client and BlobManager instances
        di_client = get_di_client()
        blob_manager_raw = get_blob_manager_raw()
        blob_manager_processed = get_blob_manager()

        # Process document from blob storage with new parameters
        result_blob_path = await document_to_markdown(
            request.course_id,
            request.section_id,
            request.file_id,
            request.document_name,
            request.document_url,
            di_client,
            blob_manager_raw=blob_manager_raw,
            blob_manager_processed=blob_manager_processed
        )

        if result_blob_path:
            logger.info(f"Document processing completed successfully: {result_blob_path}")
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            logger.error(f"Document processing failed for: {request.document_name}")
            return {
                "success": False,
                "blob_path": None
            }

    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}", exc_info=True)
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
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Video processing failed"}
    },
    tags=["Video Processing"]
)
async def process_video_file(request: ProcessVideoRequest):
    """
    Process Video File with Azure Video Indexer

    **Function Description:**
    Processes video from blob storage using Azure Video Indexer for transcription and analysis, then converts to Markdown format.

    **What to Expect:**
    • Video is processed from blob storage using Video Indexer
    • Automatic Hebrew transcription with timestamps
    • AI-powered extraction of keywords and topics
    • Subject type classification ("Humanities" or "Mathematics")
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
        "video_name": "First Lesson - Trimmed",
        "video_url": "L1_091004f349688522f773afc884451c9af6da18fb_Trim.mp4"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        logger.info(f"Starting video processing: {request.video_name}")
        logger.info(f"CourseID: {request.course_id}, SectionID: {request.section_id}, FileID: {request.file_id}")
        logger.debug(f"VideoURL: {request.video_url}")

        # Validate input parameters
        if not request.course_id or not request.section_id:
            raise HTTPException(status_code=422, detail="course_id and section_id are required")

        if not request.video_name or not request.video_url:
            raise HTTPException(status_code=422, detail="video_name and video_url are required")

        if request.file_id is None or request.file_id < 0:
            raise HTTPException(status_code=422, detail="file_id must be a non-negative integer")

        # Get shared BlobManager instances
        blob_manager_raw = get_blob_manager_raw()
        blob_manager_processed = get_blob_manager()

        # Start async video processing - returns immediately with target path
        video_processor = get_video_processor()
        result_blob_path = await video_processor.process_video_to_md(
            request.course_id,
            request.section_id,
            request.file_id,
            request.video_name,
            request.video_url,
            blob_manager_raw=blob_manager_raw,
            blob_manager_processed=blob_manager_processed
        )

        if result_blob_path:
            logger.info(f"Video processing started successfully, target path: {result_blob_path}")
            logger.info(f"Processing continues in background")
            return ProcessVideoResponse(
                success=True,
                blob_path=result_blob_path
            )
        else:
            logger.error("Video processing failed to start")
            return ProcessVideoResponse(
                success=False,
                blob_path=None
            )

    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@app.post(
    "/process/video_from_id",
    response_model=ProcessVideoResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Video processing failed"}
    },
    tags=["Video Processing"]
)
async def process_video_from_id(request: ProcessVideoFromIdRequest):
    """
    Process Video from Existing Video ID in Azure Video Indexer

    **Function Description:**
    Processes an existing video in Azure Video Indexer using its video_id to create Markdown format transcription and analysis.

    **What to Expect:**
    • Uses existing video_id from Video Indexer (no upload needed)
    • Waits for video processing completion if still in progress
    • Automatic Hebrew transcription with timestamps
    • AI-powered extraction of keywords and topics
    • Subject type classification ("Humanities" or "Mathematics")
    • Full transcript with precise timestamps
    • Saves markdown to blob storage with structure: CourseID/SectionID/Videos_md/FileID.md
    • Video name is included in the transcription

    **Request Body Example:**
    ```json
    {
        "course_id": "CS101",
        "section_id": "Section1",
        "file_id": 2,
        "video_name": "First Lesson - Trimmed",
        "video_id": "n7qnox2f7w"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created markdown file in blob storage (or None if failed)
    """
    try:
        logger.info(f"Starting video processing from video_id: {request.video_name}")
        logger.info(f"CourseID: {request.course_id}, SectionID: {request.section_id}, FileID: {request.file_id}")
        logger.debug(f"VideoID: {request.video_id}")

        # Validate input parameters
        if not request.course_id or not request.section_id:
            raise HTTPException(status_code=422, detail="course_id and section_id are required")

        if not request.video_name or not request.video_id:
            raise HTTPException(status_code=422, detail="video_name and video_id are required")

        if request.file_id is None or request.file_id < 0:
            raise HTTPException(status_code=422, detail="file_id must be a non-negative integer")

        # Get shared BlobManager instance for processed data
        blob_manager_processed = get_blob_manager()

        # Process video using existing video_id
        video_processor = get_video_processor()
        result_blob_path = await video_processor.process_video_to_md_from_id(
            video_id=request.video_id,
            course_id=request.course_id,
            section_id=request.section_id,
            file_id=request.file_id,
            video_name=request.video_name,
            blob_manager_processed=blob_manager_processed
        )

        if result_blob_path:
            logger.info(f"Video processing from ID completed successfully: {result_blob_path}")
            return ProcessVideoResponse(
                success=True,
                blob_path=result_blob_path
            )
        else:
            logger.error("Video processing from ID failed")
            return ProcessVideoResponse(
                success=False,
                blob_path=None
            )

    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        logger.error(f"Error in video processing from ID: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing from ID failed: {str(e)}")


# ================================
# INDEXING ENDPOINTS
# ================================

@app.post(
    "/insert_to_index",
    response_model=IndexResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or content type"},
        500: {"model": ErrorResponse, "description": "Indexing failed"}
    },
    tags=["Indexing"]
)
async def insert_to_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index MD Files from Blob Storage to Search Index

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

        # Add indexing task to background tasks - no await needed
        background_tasks.add_task(
            index_content_files,
            request.blob_paths,
            request.create_new_index,
            openai_client=getattr(app.state, "shared_openai_client", None),
            blob_manager=getattr(app.state, "blob_manager", None),
        )

        # Return immediately
        return {
            "message": f"Indexing started for {len(request.blob_paths)} files. Processing continues in background. Check logs for progress.",
            "create_new_index": request.create_new_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")


@app.post(
    "/delete_from_index",
    response_model=DeleteContentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Content not found"},
        500: {"model": ErrorResponse, "description": "Delete operation failed"}
    },
    tags=["Indexing"]
)
async def delete_from_index(request: DeleteContentRequest):
    """
    Delete Content from Search Index by Source ID

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
        indexer = get_content_indexer()

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
    Create Summary from Markdown File in Blob Storage

    **Function Description:**
    Generates an LLM-based summary from a Markdown file stored in blob storage with subject-specific customization.

    **What to Expect:**
    • Downloads MD file from blob storage
    • Automatically detects content type from path structure (Videos_md = video, Docs_md = document)
    • Processes the entire markdown content with subject-aware prompts
    • Returns a comprehensive summary tailored to subject type
    • Uses Azure OpenAI for intelligent summarization
    • Saves summary back to blob storage in CourseID/SectionID/file_summaries/FileID.md

    **Request Body Examples:**

    **Example 1 - Mathematics Video with Full Parameters:**
    ```json
    {
        "blob_path": "Discrete_mathematics/Section2/Videos_md/2.md",
        "subject_name": "Discrete Mathematics",
        "subject_type": "Mathematics"
    }
    ```


    **Parameters:**
    - **blob_path** (required): Path to MD file in blob storage
    - **subject_name** (optional): Name of the subject for context (e.g., "Discrete Mathematics", "Physics", "History")
    - **subject_type** (optional): Type of subject for prompt customization:
      - "Mathematics": For math, physics, computer science, engineering (includes formulas, proofs, algorithms)
      - "Humanities": For humanities, history, literature, philosophy (includes concepts, arguments, examples)

    **Content Type Detection:**
    - Files in 'Videos_md' folders → treated as video transcripts
    - Files in 'Docs_md' folders → treated as document content

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
        summarizer = get_summarizer()
        result_blob_path = await summarizer.summarize_md_file(
            request.blob_path,
            subject_name=request.subject_name,
            subject_type=request.subject_type
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


@app.post(
    "/summarize/md_files",
    response_model=SummarizeFilesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty list"},
        500: {"model": ErrorResponse, "description": "Batch summarization failed"}
    },
    tags=["Summarization"]
)
async def summarize_md_files(request: SummarizeFilesRequest, background_tasks: BackgroundTasks):
    """
    Create Summaries from Multiple Markdown Files in Blob Storage (Batch Processing)

    **Function Description:**
    Generates LLM-based summaries from multiple Markdown files stored in blob storage with asynchronous processing and queue management, similar to the insert_to_index function.

    **What to Expect:**
    • Downloads multiple MD files from blob storage in batches
    • Automatically detects content type from path structure (Videos_md = video, Docs_md = document)
    • Processes files asynchronously with controlled concurrency
    • Uses subject-aware prompts for each file
    • Returns comprehensive summaries tailored to subject type
    • Uses Azure OpenAI for intelligent summarization
    • Saves summaries back to blob storage in CourseID/SectionID/file_summaries/FileID.md
    • Processing continues in background with detailed logging

    **Request Body Examples:**

    **Example 1 - Multiple Files with Subject Parameters:**
    ```json
    {
        "blob_paths": [
            "Discrete_mathematics/Section2/Videos_md/2.md",
            "Discrete_mathematics/Section2/Docs_md/1.md",
            "Discrete_mathematics/Section2/Videos_md/3.md"
        ],
        "subject_name": "Discrete Mathematics",
        "subject_type": "Mathematics"
    }
    ```

    **Parameters:**
    - **blob_paths** (required): List of paths to MD files in blob storage
    - **subject_name** (optional): Name of the subject for context (e.g., "Discrete Mathematics", "Physics", "History")
    - **subject_type** (optional): Type of subject for prompt customization:
      - "Mathematics": For math, physics, computer science, engineering (includes formulas, proofs, algorithms)
      - "Humanities": For humanities, history, literature, philosophy (includes concepts, arguments, examples)

    **Content Type Detection:**
    - Files in 'Videos_md' folders → treated as video transcripts
    - Files in 'Docs_md' folders → treated as document content

    **Returns:**
    - success: Boolean indicating if the batch processing started successfully
    - results: Dictionary mapping original blob paths to summary paths (empty initially, populated during processing)
    - total_processed: Number of files that will be processed
    - successful: Number of files successfully processed (0 initially)
    - failed: Number of files that failed processing (0 initially)

    **Note:** This endpoint returns immediately after starting the batch process. Check logs for detailed progress and results.
    """
    try:
        # Validate blob paths
        if not request.blob_paths:
            raise HTTPException(status_code=400, detail="Blob paths list cannot be empty")

        # Check all files are MD
        md_files = []
        for blob_path in request.blob_paths:
            if blob_path.lower().endswith('.md'):
                md_files.append(blob_path)
            else:
                logger.warning(f"Skipping non-MD file: {blob_path}")

        if not md_files:
            raise HTTPException(status_code=400, detail="No MD files found in the provided list")

        logger.info(f"Starting batch summarization for {len(md_files)} MD files")
        logger.info(f"Subject: {request.subject_name} ({request.subject_type})")

        # Add summarization task to background tasks
        background_tasks.add_task(
            _batch_summarize_files,
            md_files,
            request.subject_name,
            request.subject_type
        )

        # Return immediately with initial response
        return SummarizeFilesResponse(
            success=True,
            results={},  # Will be populated during processing
            total_processed=len(md_files),
            successful=0,  # Will be updated during processing
            failed=0  # Will be updated during processing
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting batch summarization: {str(e)}")


async def _batch_summarize_files(blob_paths: List[str], subject_name: str = None, subject_type: str = None):
    """
    Background task for batch summarization
    """
    try:
        logger.info(f"Background batch summarization started for {len(blob_paths)} files")

        # Get summarizer instance
        summarizer = get_summarizer()

        # Use the new batch summarization method
        results = await summarizer.summarize_md_files(
            blob_paths=blob_paths,
            subject_name=subject_name,
            subject_type=subject_type
        )

        # Log final results
        successful = sum(1 for result in results.values() if result is not None)
        failed = len(results) - successful

        logger.info(f"Batch summarization completed!")
        logger.info(f"Total files: {len(blob_paths)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {(successful / len(blob_paths) * 100):.1f}%")

        # Log individual results
        for blob_path, summary_path in results.items():
            if summary_path:
                logger.info(f"✓ {blob_path} -> {summary_path}")
            else:
                logger.warning(f"✗ {blob_path} -> Failed")

    except Exception as e:
        logger.error(f"Error in background batch summarization: {str(e)}")


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
    Create Section Summary from Azure Storage

    **Function Description:**
    Scans all summary files in the specified blob path and creates a unified section summary with subject-specific customization.

    **What to Expect:**
    • Connects to Azure Storage
    • Scans the specified blob path for summary files
    • Processes markdown summary files (.md) from the path
    • Creates a comprehensive section-level summary tailored to subject type
    • Uses Azure OpenAI for intelligent content analysis
    • Saves section summary back to blob storage in CourseID/section_summaries/SectionID.md

    **Request Body Examples:**

    **Example 1 - Mathematics Section with Subject Parameters:**
    ```json
    {
        "full_blob_path": "Discrete_mathematics/Section2/file_summaries",
        "subject_name": "Discrete Mathematics",
        "subject_type": "Mathematics"
    }
    ```

    **Example 2 - Section with Previous Summary Context:**
    ```json
    {
        "full_blob_path": "Discrete_mathematics/Section2/file_summaries",
        "subject_name": "Discrete Mathematics",
        "subject_type": "Mathematics",
        "previous_summary_path": "Discrete_mathematics/section_summaries/Section1.md"
    }
    ```

    **Parameters:**
    - **full_blob_path** (required): Path to file_summaries folder in blob storage
    - **subject_name** (optional): Name of the subject for context
    - **subject_type** (optional): Type of subject for prompt customization ("Mathematics" or "Humanities")
    - **previous_summary_path** (optional): Path to previous section summary file for context and continuity

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        summarizer = get_summarizer()
        result_blob_path = await summarizer.summarize_section_from_blob(
            request.full_blob_path,
            subject_name=request.subject_name,
            subject_type=request.subject_type,
            previous_summary_path=request.previous_summary_path
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
    Create Complete Course Summary from Azure Storage

    **Function Description:**
    Analyzes all section summary files in the specified blob path and generates a comprehensive course-level summary with subject-specific customization.

    **What to Expect:**
    • Connects to Azure Storage
    • Scans the specified blob path for section summary files
    • Processes all section summary markdown files (.md)
    • Creates a high-level course overview and summary tailored to subject type
    • Uses Azure OpenAI for intelligent course-level analysis
    • Saves course summary back to blob storage in CourseID/course_summary.md

    **Request Body Examples:**

    **Example 1 - Mathematics Course with Subject Parameters:**
    ```json
    {
        "full_blob_path": "Discrete_mathematics/section_summaries",
        "subject_name": "Discrete Mathematics",
        "subject_type": "Mathematics"
    }
    ```

    **Parameters:**
    - **full_blob_path** (required): Path to section_summaries folder in blob storage
    - **subject_name** (optional): Name of the subject for context
    - **subject_type** (optional): Type of subject for prompt customization ("Mathematics" or "Humanities")

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created summary file in blob storage
    """
    try:
        summarizer = get_summarizer()
        result_blob_path = await summarizer.summarize_course_from_blob(
            request.full_blob_path,
            subject_name=request.subject_name,
            subject_type=request.subject_type
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
# 🔍 SUBJECT DETECTION ENDPOINTS
# ================================

@app.post(
    "/detect/subject",
    response_model=DetectSubjectResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No files found in course"},
        500: {"model": ErrorResponse, "description": "Subject detection failed"}
    },
    tags=["Subject Detection"]
)
async def detect_subject_type(request: DetectSubjectRequest):
    """
    Detect Subject Type from Course Content

    **Function Description:**
    Analyzes course content (videos and documents) to automatically determine the subject name and type using LLM analysis.

    **What the Function Does:**
    • Scans all MD files in the specified course path
    • Limits analysis to prevent token overflow (max 5 videos, 10 documents, 10K chars per file)
    • Sends content to LLM for intelligent subject identification
    • Returns both subject name and classification

    **Subject Classification:**
    - **Mathematics**: Mathematics, Physics, Computer Science, Engineering, Statistics, Logic, Algorithms
    - **Humanities**: Literature, History, Philosophy, Psychology, Sociology, Arts, Languages

    **Request Body Example:**
    ```json
    {
        "course_path": "CS101"
    }
    ```

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - subject_name: Name of the detected subject (e.g., "Computer Science" or "מדעי המחשב")
    - subject_type: Detected subject type ("Mathematics", "Humanities", or "Not detected")
    """
    try:
        logger.info(f"Starting subject type detection for course: {request.course_path}")

        # Use the shared subject detector instance
        subject_detector = get_subject_detector()
        result = await subject_detector.detect_subject_info(request.course_path)

        # Check if detection was successful
        if result["name"] == "Not detected" or result["type"] == "Not detected":
            # Check if it's because no files were found
            raise HTTPException(
                status_code=404,
                detail=f"No content files found for course: {request.course_path}"
            )

        return DetectSubjectResponse(
            success=True,
            subject_name=result["name"],
            subject_type=result["type"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"Error in subject type detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting subject type: {str(e)}"
        )


# ================================
# 📋 SYLLABUS GENERATION ENDPOINTS
# ================================

@app.post(
    "/create/syllabus",
    response_model=SummarizeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Course summary not found"},
        500: {"model": ErrorResponse, "description": "Syllabus creation failed"}
    },
    tags=["Syllabus Generation"]
)
async def create_syllabus_from_course_summary(request: CreateSyllabusRequest):
    """
    Create Academic Syllabus from Course Summary

    **Function Description:**
    Generates a structured academic syllabus from a course summary file using AI analysis with subject-specific customization.

    **What to Expect:**
    • Downloads course summary file from blob storage
    • Analyzes the complete course content using Azure OpenAI
    • Creates a professional, structured syllabus document
    • Tailors content based on subject type (mathematical vs. humanities)
    • Saves syllabus back to blob storage in CourseID/syllabus.md

    **Request Body Examples:**

    **Example 1 - Medieval History Course:**
    ```json
    {
        "full_blob_path": "Intro_to_medieval_history/course_summary.md",
        "subject_name": "Introduction to Medieval History",
        "subject_type": "Humanities"
    }
    ```

    **Parameters:**
    - **full_blob_path** (required): Path to course summary file (must end with '/course_summary.md')
    - **subject_name** (optional): Name of the subject for context
    - **subject_type** (optional): Type of subject for prompt customization:
      - "Mathematics": For math, physics, computer science, engineering
      - "Humanities": For humanities, history, literature, philosophy

    **Returns:**
    - success: Boolean indicating if the operation was successful
    - blob_path: Path to the created syllabus file in blob storage (CourseID/syllabus.md)
    """
    try:
        logger.info(f"Starting syllabus creation from: {request.full_blob_path}")
        logger.info(f"Subject: {request.subject_name} ({request.subject_type})")

        # Validate path format
        if not request.full_blob_path.endswith('/course_summary.md'):
            raise HTTPException(
                status_code=400,
                detail="Invalid path format. Path must end with '/course_summary.md'"
            )

        # Use syllabus generator
        syllabus_generator = get_syllabus_generator()
        result_blob_path = await syllabus_generator.create_syllabus_from_course_summary(
            request.full_blob_path,
            subject_name=request.subject_name,
            subject_type=request.subject_type
        )

        if result_blob_path:
            logger.info(f"Syllabus created successfully: {result_blob_path}")
            return {
                "success": True,
                "blob_path": result_blob_path
            }
        else:
            logger.error(f"Failed to create syllabus from: {request.full_blob_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Course summary not found or failed to process: {request.full_blob_path}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in syllabus creation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating syllabus: {str(e)}"
        )


# ================================
# 🚀 SERVER STARTUP
# ================================

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    logger.info("API documentation available at: http://localhost:8080/docs")
    logger.info("Home page: http://localhost:8080/")
    logger.info("Stop server: Ctrl+C")

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8080,
        log_level="info",
        reload=True
    )
