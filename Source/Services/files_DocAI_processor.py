import os
import asyncio
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from Source.Services.blob_manager import BlobManager
from Config.config import AZURE_FORM_RECOGNIZER_KEY, AZURE_FORM_RECOGNIZER_ENDPOINT
from Config.logging_config import setup_logging
logger = setup_logging()

async def process_single_document(file_path: str) -> str | None:
    """
    Accepts *any* document path (.doc, .docx, .pdf, .pptx, .png, …)
    and returns a Markdown string (or None if it failed).
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    work_path = file_path
    logger.info(f"Processing {work_path} → MD File")

    # Create client per request to avoid connection issues
    client = None
    try:
        # Create new client for this request
        client = DocumentIntelligenceClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )

        # Use Azure Document Intelligence with proper async context manager
        async with client:
            with open(work_path, "rb") as f:
                poller = await client.begin_analyze_document(
                    "prebuilt-layout",
                    f,
                    content_type=None,  # let the service detect type
                    output_content_format="markdown"
                )
            result = await poller.result()
            md = result.content
            return md

    except Exception as e:
        logger.error(f"Error in Azure DI for {work_path}: {e}")
        return None
    finally:
        # Ensure client is properly closed
        if client:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")


async def process_document_from_memory(file_bytes: bytes) -> str | None:
    """
    Processes document bytes directly using Azure Document Intelligence.

    Args:
        file_bytes: The document content as bytes

    Returns:
        Markdown string or None if failed
    """
    # Create client per request to avoid connection issues
    client = None
    try:
        logger.info(f"Processing document from memory ({len(file_bytes)} bytes) → MD")

        # Create BytesIO object from bytes
        file_buffer = BytesIO(file_bytes)

        # Create new client for this request
        client = DocumentIntelligenceClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )

        # Use Azure Document Intelligence with proper async context manager
        async with client:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                file_buffer,
                content_type=None,  # let the service detect type
                output_content_format="markdown"
            )
            result = await poller.result()
            md = result.content
            return md

    except Exception as e:
        logger.error(f"Error in Azure DI for memory buffer: {e}")
        return None
    finally:
        # Ensure client is properly closed
        if client:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")


async def document_to_markdown(course_id: str, section_id: str, file_id: int, document_name: str, document_url: str) -> str | None:
    """
    Streamlined: Downloads document from blob → processes in memory → uploads markdown.
    No temp files or disk I/O.

    Args:
        course_id: Course identifier
        section_id: Section identifier
        file_id: File identifier
        document_name: Document name (will be included in transcription)
        document_url: File path in blob storage (e.g., "Section1/Raw-data/Docs/Ex5Sol.pdf")

    Returns:
        File path in blob storage or None if failed
    """
    # Create blob managers - one for reading from raw-data and one for writing to processeddata
    blob_manager_read = BlobManager(container_name="raw-data")
    blob_manager_write = BlobManager(container_name="processeddata")

    # Check file extension
    supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = os.path.splitext(document_url)[1].lower()

    if file_ext not in supported_extensions:
        logger.info(f"Unsupported file type: {document_url}")
        return None

    logger.info(f"Downloading file from raw-data container: {document_url}")

    # Step 1: Download blob directly to memory from raw-data container
    file_bytes = await blob_manager_read.download_to_memory(document_url)
    if not file_bytes:
        logger.info(f"Failed to download file to memory from raw-data container: {document_url}")
        return None

    logger.info(f"Processing file in memory: {document_name}")

    # Step 2: Process document directly from memory
    md_content = await process_document_from_memory(file_bytes)
    if not md_content:
        logger.info(f"Failed to process file: {document_name}")
        return None

    # Add document name to the beginning of the transcription
    enhanced_md_content = f"# {document_name}\n\n{md_content}"

    # Step 3: Upload markdown directly to processeddata container
    # Create path according to structure: CourseID/SectionID/Docs_md/FileID.md
    target_blob_path = f"{course_id}/{section_id}/Docs_md/{file_id}.md"

    logger.info(f"Uploading markdown to processeddata container: {target_blob_path}")

    success = await blob_manager_write.upload_text_to_blob(
        text_content=enhanced_md_content,
        blob_name=target_blob_path
    )

    if success:
        logger.info(f"File uploaded successfully to processeddata container: {target_blob_path}")
        return target_blob_path
    else:
        logger.info(f"Failed to upload file to processeddata container")
        return None



async def main():
    # Process a single document from blob storage with new parameters
    course_id = "CS101"
    section_id = "Section1"
    file_id = 1
    document_name = "בדידה תרגול 02"
    # document_url = "Section1/Raw-data/Docs/bdida_tirgul_02.pdf"
    # document_url = "Section1/Raw-data/Docs/bdida_tirgul_02.pdf"
    document_url = "bdida_tirgul_02.pdf"

    logger.info(f"Processing file: {document_name}")
    logger.info(f"CourseID: {course_id}, SectionID: {section_id}, FileID: {file_id}")
    logger.info(f"DocumentURL: {document_url}")

    # Process the document (now with await)
    result = await document_to_markdown(course_id, section_id, file_id, document_name, document_url)

    if result:
        logger.info(f"\nFile processed successfully: {result}")
    else:
        logger.info(f"\nFailed to process file: {document_name}")


if __name__ == "__main__":
    asyncio.run(main())
