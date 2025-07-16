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


def document_to_markdown(blob_file_path: str) -> str | None:
    """
    Streamlined: Downloads document from blob → processes in memory → uploads markdown.
    No temp files or disk I/O.

    Args:
        blob_file_path: נתיב הקובץ ב-blob storage (למשל: "Section1/Raw-data/Docs/Ex5Sol.pdf")

    Returns:
        נתיב הקובץ ב-blob storage או None אם נכשל
    """
    # יצירת מנהל blob
    blob_manager = BlobManager()

    # בדיקת סיומת הקובץ
    supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = os.path.splitext(blob_file_path)[1].lower()

    if file_ext not in supported_extensions:
        print(f"❌ סוג קובץ לא נתמך: {blob_file_path}")
        return None

    # חילוץ מבנה התיקיות
    # למשל: "Section1/Raw-data/Docs/Ex5Sol.pdf" -> "Section1"
    path_parts = blob_file_path.split('/')

    # מחפש את האינדקס של "Raw-data"
    try:
        raw_data_index = path_parts.index("Raw-data")
        # לוקח את כל החלקים לפני "Raw-data"
        base_path = '/'.join(path_parts[:raw_data_index]) if raw_data_index > 0 else ""
    except ValueError:
        # אם אין "Raw-data" בנתיב, נשתמש בנתיב הבסיסי
        base_path = ""

    # קבלת שם הקובץ
    filename = os.path.basename(blob_file_path)
    base_name = os.path.splitext(filename)[0]

    print(f"🌐 מוריד קובץ מ-blob לזיכרון: {blob_file_path}")

    # Step 1: Download blob directly to memory
    file_bytes = blob_manager.download_to_memory(blob_file_path)
    if not file_bytes:
        print(f"❌ נכשלה הורדת הקובץ לזיכרון: {blob_file_path}")
        return None

    print(f"🔄 מעבד קובץ בזיכרון: {filename}")

    # Step 2: Process document directly from memory
    md_content = process_document_from_memory(file_bytes)
    if not md_content:
        print(f"❌ נכשל עיבוד הקובץ: {filename}")
        return None

    # Step 3: Upload markdown directly to blob storage
    md_filename = f"{base_name}.md"

    if base_path:
        target_blob_path = f"{base_path}/Processed-data/Docs_md/{md_filename}"
    else:
        target_blob_path = f"Processed-data/Docs_md/{md_filename}"

    print(f"📤 מעלה markdown ל-blob: {target_blob_path}")

    success = blob_manager.upload_text_to_blob(
        text_content=md_content,
        blob_name=target_blob_path
    )

    if success:
        print(f"✅ הקובץ הועלה בהצלחה ל-blob: {target_blob_path}")
        return target_blob_path
    else:
        print(f"❌ נכשלה העלאת הקובץ ל-blob storage")
        return None


if __name__ == "__main__":
    # Process a single document from blob storage with folder structure
    # blob_file_path = "Section1/Raw-data/Docs/Ex5Sol.pdf"
    blob_file_path = "Section1/Raw-data/Docs/bdida_tirgul_02.pdf"

    print(f"🧪 מעבד קובץ: {blob_file_path}")

    # Process the document
    result = document_to_markdown(blob_file_path)

    if result:
        print(f"\n🎉 הקובץ עובד בהצלחה: {result}")
        print(f"📁 הקובץ נשמר עם שמירה על מבנה התיקיות")
    else:
        print(f"\n❌ נכשל עיבוד הקובץ: {blob_file_path}")

