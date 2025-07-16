"""
Unified Content Indexer - מאחד וידאו ומסמכים באינדקס אחד
מקבל תוכן מסוגים שונים ומאחסן אותם באינדקס משותף עם שדות גמישים

Core workflow:
1. Take content from videos OR documents
2. Chunk the content appropriately
3. Generate embeddings for chunks
4. Push chunks to unified Azure AI Search index with flexible schema
"""

import uuid
import tiktoken
import re
import os
from typing import List, Dict
from datetime import datetime, timezone
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile,
    SearchField, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, SemanticSearch
)
from openai import AzureOpenAI
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from Source.Services.blob_manager import BlobManager
from Config.config import (
    SEARCH_SERVICE_NAME, SEARCH_API_KEY,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_MODEL,
    INDEX_NAME
)


class UnifiedContentIndexer:
    """
    אינדקסר מאוחד לתוכן מסוגים שונים - וידאו ומסמכים
    משתמש בסכמה גמישה שמתאימה לשני הסוגים
    """

    def __init__(self):
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.index_client = SearchIndexClient(self.search_endpoint, self.credential)
        self.index_name = INDEX_NAME

        # Configure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def create_index(self, create_new: bool = True) -> bool:
        """Create the unified content chunks index with flexible schema

        Args:
            create_new: If True (default), delete existing index and create new one.
                       If False, use existing index or create if doesn't exist.
        """
        print(f"🔧 Setting up unified index: {self.index_name}")

        try:
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                if create_new:
                    print(f"🗑️ Deleting existing index: {self.index_name}")
                    self.index_client.delete_index(self.index_name)
                    print(f"📝 Creating new unified index: {self.index_name}")
                else:
                    print(f"✅ Using existing unified index: {self.index_name}")
                    return True
            except ResourceNotFoundError:
                print(f"📝 Creating new unified index: {self.index_name}")

            # Vector search configuration
            hnsw_algo = HnswAlgorithmConfiguration(name="my-hnsw-config")
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="my-vector-profile",
                        algorithm_configuration_name="my-hnsw-config"
                    )
                ],
                algorithms=[hnsw_algo]
            )

            # Unified fields schema - supports both videos and documents
            fields = [
                # Core identification
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                # "video" or "document"
                SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                # video_id or document_id
                SimpleField(name="source_name", type=SearchFieldDataType.String, filterable=True),
                # video_name or document_name

                # Content
                SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="my-vector-profile",
                ),

                # Chunk information
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),

                # Video-specific fields (nullable for documents)
                SimpleField(name="start_seconds", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="start_time", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="end_seconds", type=SearchFieldDataType.Double, filterable=True, sortable=True),

                # Document-specific fields (nullable for videos)
                SearchableField(name="section_title", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),

                # Common metadata
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True,
                            sortable=True),
                SearchableField(name="keywords", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SearchableField(name="topics", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
            ]

            # Semantic search configuration
            semantic_config = SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")],
                    keywords_fields=[
                        SemanticField(field_name="source_name"),
                        SemanticField(field_name="section_title"),
                        SemanticField(field_name="keywords"),
                        SemanticField(field_name="topics")
                    ]
                )
            )

            semantic_search = SemanticSearch(configurations=[semantic_config])

            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )

            self.index_client.create_or_update_index(index)
            print("✅ Unified index created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating unified index: {e}")
            return False

    def embed_texts_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  🔄 Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

            try:
                response = self.openai_client.embeddings.create(
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"❌ Error generating embeddings: {e}")
                embeddings.extend([[] for _ in batch])

        return embeddings

    def chunk_by_tokens(self, text: str, max_tokens: int = 2000) -> List[str]:
        """Split text by tokens"""
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, errors="replace")
            chunks.append(chunk_text)

        return chunks

    def _process_video_segments_to_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Convert video transcript segments to searchable chunks"""
        chunks = []
        chunk_idx = 0

        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            # Check if segment needs chunking
            token_count = len(self.tokenizer.encode(text))

            if token_count <= 2000:
                # Segment is fine as is
                chunk = {
                    "text": text,
                    "chunk_index": chunk_idx,
                    "start_seconds": segment.get("start_seconds", 0),
                    "start_time": segment.get("start_time", "00:00:00"),
                    "end_seconds": segment.get("end_seconds", segment.get("start_seconds", 0) + 5.0)
                }
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Split large segment
                print(f"  📝 Splitting video segment ({token_count} tokens)")
                text_chunks = self.chunk_by_tokens(text)

                start_seconds = segment.get("start_seconds", 0)
                duration_per_chunk = 2.0  # Default 2 seconds per chunk

                for i, chunk_text in enumerate(text_chunks):
                    chunk_start = start_seconds + (i * duration_per_chunk)
                    chunk_end = chunk_start + duration_per_chunk

                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_idx,
                        "start_seconds": chunk_start,
                        "start_time": segment.get("start_time", "00:00:00"),
                        "end_seconds": chunk_end
                    }
                    chunks.append(chunk)
                    chunk_idx += 1

        return chunks

    def _process_document_to_chunks(self, markdown_content: str) -> List[Dict]:
        """Convert document markdown to searchable chunks"""
        # Split by headers to preserve sections
        sections = re.split(r'\n#+\s+', markdown_content)
        chunks = []
        chunk_idx = 0

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Extract section title and body
            lines = section.strip().split('\n', 1)
            if i == 0 and not section.startswith('#'):
                # First section might not have a header
                section_title = ""
                section_body = section.strip()
            else:
                section_title = lines[0].strip() if lines else ""
                section_body = lines[1].strip() if len(lines) > 1 else ""

            if not section_body:
                continue

            # Check if section needs chunking
            token_count = len(self.tokenizer.encode(section_body))

            if token_count <= 2000:
                # Section is fine as is
                chunks.append({
                    "text": section_body,
                    "chunk_index": chunk_idx,
                    "section_title": section_title
                })
                chunk_idx += 1
            else:
                # Split large section
                print(f"  📝 Splitting document section ({token_count} tokens)")
                text_chunks = self.chunk_by_tokens(section_body)

                for chunk_text in text_chunks:
                    chunks.append({
                        "text": chunk_text,
                        "chunk_index": chunk_idx,
                        "section_title": section_title
                    })
                    chunk_idx += 1

        return chunks

    def get_stats(self) -> Dict:
        """Get statistics for the unified index"""
        try:
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)

            # Total count
            results = search_client.search("*", include_total_count=True, top=0)
            total_docs = results.get_count()

            # Count by content type
            video_results = search_client.search("*", filter="content_type eq 'video'", include_total_count=True, top=0)
            video_count = video_results.get_count()

            doc_results = search_client.search("*", filter="content_type eq 'document'", include_total_count=True,
                                               top=0)
            doc_count = doc_results.get_count()

            stats = {
                "total_chunks": total_docs,
                "video_chunks": video_count,
                "document_chunks": doc_count
            }

            print(f"📊 Unified Index Statistics:")
            print(f"  📄 Total chunks: {total_docs}")
            print(f"  🎥 Video chunks: {video_count}")
            print(f"  📝 Document chunks: {doc_count}")

            return stats

        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}


def _detect_content_type_from_path(blob_path: str) -> str:
    """
    זיהוי סוג התוכן לפי נתיב הקובץ
    מחזיר 'video' אם הנתיב מכיל 'Videos_md' או 'document' אם מכיל 'Docs_md'
    """
    if "Videos_md" in blob_path.lower():
        return "video"
    elif "Docs_md" in blob_path.lower():
        return "document"
    else:
        # ברירת מחדל - ננסה לזהות לפי סיומת
        if blob_path.lower().endswith('.md'):
            return "document"  # ברירת מחדל למסמכים
        return "unknown"



def parse_video_md_from_blob(blob_path: str, blob_manager: BlobManager) -> Dict:
    """
    Parse video MD file from blob storage and convert to structured data format expected by indexer
    """

    print(f"📖 Reading video MD file from blob: {blob_path}")

    # Download content from blob to memory
    file_bytes = blob_manager.download_to_memory(blob_path)
    if not file_bytes:
        raise Exception(f"Failed to download blob: {blob_path}")

    content = file_bytes.decode('utf-8')

    # Extract video information - support both Hebrew and English formats
    video_id_match = re.search(r'\*\*(?:Video ID|מזהה וידאו)\*\*: (\w+)', content)
    video_id = video_id_match.group(1) if video_id_match else "unknown"

    duration_match = re.search(r'\*\*(?:Duration|משך זמן)\*\*: ([\d:]+)', content)
    duration = duration_match.group(1) if duration_match else "N/A"

    language_match = re.search(r'\*\*(?:Language|שפה)\*\*: ([\w-]+)', content)
    language = language_match.group(1) if language_match else "he-IL"

    speakers_match = re.search(r'\*\*(?:Speakers|דוברים)\*\*: (.+)', content)
    speakers = [speakers_match.group(1)] if speakers_match else ["מרצה"]

    created_match = re.search(r'\*\*(?:Created|תאריך יצירה)\*\*: (.+)', content)
    created_date = created_match.group(1) if created_match else datetime.now().isoformat()

    # Extract keywords
    keywords_match = re.search(r'## 🔍 Keywords\n`(.+)`', content)
    keywords = []
    if keywords_match:
        keywords_text = keywords_match.group(1)
        keywords = [kw.strip() for kw in keywords_text.split('`,') if kw.strip()]
        # Clean up the last keyword
        if keywords:
            keywords[-1] = keywords[-1].rstrip('`')

    # Extract topics
    topics_match = re.search(r'## 🏷️ Topics\n`(.+)`', content)
    topics = []
    if topics_match:
        topics_text = topics_match.group(1)
        topics = [topic.strip() for topic in topics_text.split('`,') if topic.strip()]
        # Clean up the last topic
        if topics:
            topics[-1] = topics[-1].rstrip('`')

    # Extract transcript segments with timestamps
    transcript_segments = []
    timestamp_pattern = r'\*\*\[([^\]]+)\]\*\* (.+)'

    # Find all timestamp entries
    timestamp_matches = re.findall(timestamp_pattern, content)

    for i, (timestamp, text) in enumerate(timestamp_matches):
        # Convert timestamp to seconds
        start_seconds = convert_timestamp_to_seconds(timestamp)

        segment = {
            "text": text.strip(),
            "start_time": timestamp,
            "start_seconds": start_seconds,
            "end_seconds": start_seconds + 5.0,  # Default 5 seconds duration
            "confidence": 0.9
        }
        transcript_segments.append(segment)

    # Create structured data with better naming
    # Extract filename for better identification
    filename = os.path.basename(blob_path)
    base_filename = os.path.splitext(filename)[0]

    # Create a more descriptive name
    if video_id != "unknown":
        video_name = f"Video {video_id} ({base_filename})"
    else:
        video_name = f"Video {base_filename}"


    # Create structured data
    structured_data = {
        "id": video_id,
        "name": video_name,
        "duration": duration,
        "language": language,
        "speakers": speakers,
        "keywords": keywords,
        "topics": topics,
        "created_date": created_date,
        "transcript_segments": transcript_segments
    }

    print(f"✅ Parsed video MD file:")
    print(f"  - Video ID: {video_id}")
    print(f"  - Duration: {duration}")
    print(f"  - Segments: {len(transcript_segments)}")
    print(f"  - Keywords: {len(keywords)}")
    print(f"  - Topics: {len(topics)}")

    return structured_data

def convert_timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp like '0:00:01.03' to seconds"""
    try:
        # Handle format like "0:00:01.03"
        parts = timestamp.split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(timestamp)
    except:
        return 0.0


def parse_document_md_from_blob(blob_path: str, blob_manager: BlobManager) -> Dict:
    """
    Parse document MD file from blob storage and convert to document data format expected by indexer
    """
    print(f"📖 Reading document MD file from blob: {blob_path}")

    # Download content from blob to memory
    file_bytes = blob_manager.download_to_memory(blob_path)
    if not file_bytes:
        raise Exception(f"Failed to download blob: {blob_path}")

    content = file_bytes.decode('utf-8')

    # Extract document info from blob path
    base_name = os.path.splitext(os.path.basename(blob_path))[0]

    # Create document data
    document_data = {
        "id": base_name,
        "name": base_name.replace('_', ' ').replace('-', ' '),
        "content": content,
        "type": "markdown"
    }

    print(f"✅ Parsed document MD file:")
    print(f"  - Document ID: {document_data['id']}")
    print(f"  - Document Name: {document_data['name']}")
    print(f"  - Content Length: {len(content)} characters")

    return document_data


def index_content_files(blob_paths: List[str], create_new_index: bool = False) -> str:
    """
    אינדוקס קבצי MD מ-blob storage לאינדקס מאוחד.
    מזהה אוטומטית את סוג הקובץ (וידאו/מסמך) לפי ה-path.

    Args:
        blob_paths: רשימת נתיבי blob של קבצי MD (למשל: ["Videos_md/video.md", "Docs_md/doc.md"])
        create_new_index: האם ליצור אינדקס חדש (True) או להוסיף לאינדקס קיים (False)

    Returns:
        הודעת סיכום עם מספר הקבצים שהועלו לאינדקס
    """
    indexer = UnifiedContentIndexer()
    blob_manager = BlobManager()

    # צור/אתחל את האינדקס - תמיד משתמש ב INDEX_NAME מהקונפיג
    if not indexer.create_index(create_new=create_new_index):
        return "❌ נכשל ביצירת האינדקס"

    all_docs = []
    processed_videos = 0
    processed_documents = 0
    skipped_files = 0

    print(f"📁 מעבד {len(blob_paths)} קבצי MD מ-blob storage...")

    for blob_path in blob_paths:
        try:
            print(f"🔄 מעבד קובץ: {blob_path}")

            # זיהוי סוג הקובץ מתוך ה-path
            content_type = _detect_content_type_from_path(blob_path)
            print(f"  📋 זוהה כסוג: {content_type}")

            if content_type == "video":
                # עיבוד קובץ וידאו
                video_data = parse_video_md_from_blob(blob_path, blob_manager)
                segments = video_data.get("transcript_segments", [])
                if not segments:
                    print(f"⚠️ קובץ {blob_path} לא מכיל תמלול, מדלגים.")
                    skipped_files += 1
                    continue

                # פיצול התמלול לחתיכות
                chunks = indexer._process_video_segments_to_chunks(segments)
                texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
                if not texts:
                    skipped_files += 1
                    continue

                # יצירת אמבדינגים
                embeddings = indexer.embed_texts_batch(texts)

                # בניית מסמכי אינדקס עבור כל חתיכה
                keywords_str = ", ".join(video_data.get("keywords", []))
                topics_str = ", ".join(video_data.get("topics", []))
                for chunk, embedding in zip(chunks, embeddings):
                    if not embedding:
                        continue
                    doc = {
                        "id": str(uuid.uuid4()),
                        "content_type": "video",
                        "source_id": video_data.get("id", "unknown"),
                        "source_name": video_data.get("name", "Unknown Video"),
                        "text": chunk.get("text", ""),
                        "vector": embedding,
                        "chunk_index": chunk.get("chunk_index", 0),
                        # שדות ייחודיים לוידאו
                        "start_seconds": chunk.get("start_seconds", 0.0),
                        "start_time": chunk.get("start_time", "00:00:00"),
                        "end_seconds": chunk.get("end_seconds", chunk.get("start_seconds", 0.0) + 5.0),
                        # שדות מסמך (לא רלוונטי לוידאו)
                        "section_title": None,
                        "document_type": None,
                        # Meta data משותף
                        "created_date": datetime.now(timezone.utc),
                        "keywords": keywords_str,
                        "topics": topics_str,
                    }
                    all_docs.append(doc)
                processed_videos += 1

            elif content_type == "document":
                # עיבוד קובץ מסמך
                doc_data = parse_document_md_from_blob(blob_path, blob_manager)
                markdown_content = doc_data.get("content", "")
                if not markdown_content:
                    print(f"⚠️ קובץ {blob_path} ריק או לא נטען, מדלגים.")
                    skipped_files += 1
                    continue

                # פיצול תוכן המסמך לחתיכות
                chunks = indexer._process_document_to_chunks(markdown_content)
                texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
                if not texts:
                    skipped_files += 1
                    continue

                # יצירת אמבדינגים
                embeddings = indexer.embed_texts_batch(texts)

                # בניית מסמכי אינדקס עבור כל חתיכה
                for chunk, embedding in zip(chunks, embeddings):
                    if not embedding:
                        continue
                    doc = {
                        "id": str(uuid.uuid4()),
                        "content_type": "document",
                        "source_id": doc_data.get("id", "unknown"),
                        "source_name": doc_data.get("name", "Unknown Document"),
                        "text": chunk.get("text", ""),
                        "vector": embedding,
                        "chunk_index": chunk.get("chunk_index", 0),
                        # שדות וידאו (לא רלוונטי למסמך)
                        "start_seconds": None,
                        "start_time": None,
                        "end_seconds": None,
                        # שדות ייחודיים למסמכים
                        "section_title": chunk.get("section_title", ""),
                        "document_type": doc_data.get("type", "markdown"),
                        # Meta data משותף
                        "created_date": datetime.now(timezone.utc),
                        "keywords": "",
                        "topics": "",
                    }
                    all_docs.append(doc)
                processed_documents += 1

            else:
                print(f"❌ לא ניתן לזהות סוג קובץ עבור: {blob_path}")
                skipped_files += 1
                continue

        except Exception as e:
            print(f"❌ שגיאה בעיבוד הקובץ {blob_path}: {e}")
            skipped_files += 1
            continue

    # הצג סיכום עיבוד
    print(f"\n📊 סיכום עיבוד:")
    print(f"  🎥 קבצי וידאו שעובדו: {processed_videos}")
    print(f"  📝 קבצי מסמכים שעובדו: {processed_documents}")
    print(f"  ⚠️ קבצים שדולגו: {skipped_files}")
    print(f"  📄 סה״כ chunks שנוצרו: {len(all_docs)}")

    # העלאה לאינדקס - משתמש ב INDEX_NAME מהקונפיג
    if all_docs:
        try:
            search_client = SearchClient(indexer.search_endpoint, INDEX_NAME, indexer.credential)
            results = search_client.upload_documents(all_docs)
            succeeded = sum(1 for r in results if r.succeeded)

            # הצג סטטיסטיקות
            indexer.get_stats()

            return f"✅ הועלו בהצלחה {succeeded} chunks מתוך {processed_videos + processed_documents} קבצים לאינדקס {INDEX_NAME}. דולגו {skipped_files} קבצים."

        except Exception as e:
            return f"❌ שגיאה בהעלאת המסמכים לאינדקס: {e}"
    else:
        return "⚠️ לא נמצאו מסמכים להעלאה (יתכן שכל הקבצים היו ריקים)."


def main():
    """Main function - demonstrates usage with automatic type detection"""
    print("🚀 Unified Content Indexer - Videos + Documents")
    print("=" * 60)

    print("\n🎯 יצירת אינדקס מאוחד עם זיהוי אוטומטי של סוג הקובץ")

    # Define blob paths to process - type will be auto-detected from path
    blob_paths = [
        "CS101/Section1/Videos_md/2.md",
        "CS101/Section1/Docs_md/1.md",
    ]

    result = index_content_files(blob_paths, create_new_index=True)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
