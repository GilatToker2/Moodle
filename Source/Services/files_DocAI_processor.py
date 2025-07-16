import os
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from Source.Services.blob_manager import BlobManager

client = DocumentIntelligenceClient(
    endpoint="https://dai-moodle-eastus2.cognitiveservices.azure.com/", credential=AzureKeyCredential(
        "BiqgEbh4itSZn93QSWz4OdaFZlQfFj5dT3N8YaYBh5bKUq6bUxO5JQQJ99BGACHYHv6XJ3w3AAALACOGqlZY")
)


def process_single_document(file_path: str) -> str | None:
    """
    Accepts *any* document path (.doc, .docx, .pdf, .pptx, .png, …)
    and returns a Markdown string (or None if it failed).
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    work_path = file_path
    print(f"🔍 Processing {work_path} → MD File")

    try:
        # use Azure Document Intelligence
        with open(work_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                f,
                content_type=None,  # let the service detect type
                output_content_format="markdown"
            )
        result = poller.result()
        md = result.content
        return md

    except Exception as e:
        print(f"❌ Error in Azure DI for {work_path}: {e}")
        return None


def process_document_from_memory(file_bytes: bytes) -> str | None:
    """
    Processes document bytes directly using Azure Document Intelligence.

    Args:
        file_bytes: The document content as bytes

    Returns:
        Markdown string or None if failed
    """
    try:
        print(f"🔍 Processing document from memory ({len(file_bytes)} bytes) → MD")

        # Create BytesIO object from bytes
        file_buffer = BytesIO(file_bytes)

        # Use Azure Document Intelligence directly with the buffer
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            file_buffer,
            content_type=None,  # let the service detect type
            output_content_format="markdown"
        )
        result = poller.result()
        md = result.content
        return md

    except Exception as e:
        print(f"❌ Error in Azure DI for memory buffer: {e}")
        return None


def document_to_markdown(course_id: str, section_id: str, file_id: int, document_name: str, document_url: str) -> str | None:
    """
    Streamlined: Downloads document from blob → processes in memory → uploads markdown.
    No temp files or disk I/O.

    Args:
        course_id: מזהה הקורס
        section_id: מזהה הסקציה
        file_id: מזהה הקובץ
        document_name: שם המסמך (ייכנס לתמלול)
        document_url: נתיב הקובץ ב-blob storage (למשל: "Section1/Raw-data/Docs/Ex5Sol.pdf")

    Returns:
        נתיב הקובץ ב-blob storage או None אם נכשל
    """
    # יצירת מנהלי blob - אחד לקריאה מ-raw-data ואחד לכתיבה ל-processeddata
    blob_manager_read = BlobManager(container_name="raw-data")
    blob_manager_write = BlobManager(container_name="processeddata")

    # בדיקת סיומת הקובץ
    supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = os.path.splitext(document_url)[1].lower()

    if file_ext not in supported_extensions:
        print(f"❌ סוג קובץ לא נתמך: {document_url}")
        return None

    print(f"🌐 מוריד קובץ מקונטיינר raw-data: {document_url}")

    # Step 1: Download blob directly to memory from raw-data container
    file_bytes = blob_manager_read.download_to_memory(document_url)
    if not file_bytes:
        print(f"❌ נכשלה הורדת הקובץ לזיכרון מקונטיינר raw-data: {document_url}")
        return None

    print(f"🔄 מעבד קובץ בזיכרון: {document_name}")

    # Step 2: Process document directly from memory
    md_content = process_document_from_memory(file_bytes)
    if not md_content:
        print(f"❌ נכשל עיבוד הקובץ: {document_name}")
        return None

    # הוספת שם המסמך לתחילת התמלול
    enhanced_md_content = f"# {document_name}\n\n{md_content}"

    # Step 3: Upload markdown directly to processeddata container
    # יצירת נתיב לפי המבנה: CourseID/SectionID/Docs_md/FileID.md
    target_blob_path = f"{course_id}/{section_id}/Docs_md/{file_id}.md"

    print(f"📤 מעלה markdown לקונטיינר processeddata: {target_blob_path}")

    success = blob_manager_write.upload_text_to_blob(
        text_content=enhanced_md_content,
        blob_name=target_blob_path
    )

    if success:
        print(f"✅ הקובץ הועלה בהצלחה לקונטיינר processeddata: {target_blob_path}")
        return target_blob_path
    else:
        print(f"❌ נכשלה העלאת הקובץ לקונטיינר processeddata")
        return None


if __name__ == "__main__":
    # Process a single document from blob storage with new parameters
    course_id = "CS101"
    section_id = "Section1"
    file_id = 1
    document_name = "בדידה תרגול 02"
    # document_url = "Section1/Raw-data/Docs/bdida_tirgul_02.pdf"
    document_url = "bdida_tirgul_02.pdf"

    print(f"🧪 מעבד קובץ: {document_name}")
    print(f"📍 CourseID: {course_id}, SectionID: {section_id}, FileID: {file_id}")
    print(f"🔗 DocumentURL: {document_url}")

    # Process the document
    result = document_to_markdown(course_id, section_id, file_id, document_name, document_url)

    if result:
        print(f"\n🎉 הקובץ עובד בהצלחה: {result}")
    else:
        print(f"\n❌ נכשל עיבוד הקובץ: {document_name}")

