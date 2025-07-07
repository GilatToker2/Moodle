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
import openai
import tiktoken
import re
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, VectorSearchAlgorithmConfiguration, VectorSearchProfile,
    SearchField, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, SemanticSearch
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from config import (
    SEARCH_SERVICE_NAME, SEARCH_API_KEY,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_MODEL
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
        self.index_name = "unified-content-chunks"

        # Configure OpenAI client
        from openai import AzureOpenAI
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
            from azure.search.documents.indexes.models import HnswAlgorithmConfiguration

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

    def index_video_content(self, structured_data: Dict) -> bool:
        """
        Index video content in the unified index

        Args:
            structured_data: Video data from processing (with transcript_segments)
        """
        video_id = structured_data.get("id", "unknown")
        video_name = structured_data.get("name", "Unknown Video")

        print(f"📤 Indexing video content: {video_name} (ID: {video_id})")

        try:
            # Get transcript segments
            segments = structured_data.get("transcript_segments", [])
            if not segments:
                print("❌ No transcript segments found")
                return False

            print(f"  📊 Processing {len(segments)} video segments")

            # Convert segments to chunks
            chunks = self._process_video_segments_to_chunks(segments)
            print(f"  📊 Created {len(chunks)} video chunks")

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
            if not texts:
                print("❌ No valid text found")
                return False

            # Generate embeddings
            print(f"  🔄 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embed_texts_batch(texts)

            # Prepare documents for unified index
            docs = []
            keywords_str = ", ".join(structured_data.get("keywords", []))
            topics_str = ", ".join(structured_data.get("topics", []))

            for chunk, embedding in zip(chunks, embeddings):
                if not embedding:
                    continue

                doc = {
                    "id": str(uuid.uuid4()),
                    "content_type": "video",
                    "source_id": video_id,
                    "source_name": video_name,
                    "text": chunk.get("text", ""),
                    "vector": embedding,
                    "chunk_index": chunk.get("chunk_index", 0),

                    # Video-specific fields
                    "start_seconds": chunk.get("start_seconds", 0),
                    "start_time": chunk.get("start_time", "00:00:00"),
                    "end_seconds": chunk.get("end_seconds"),

                    # Document-specific fields (null for videos)
                    "section_title": None,
                    "document_type": None,

                    # Common metadata
                    "created_date": datetime.now(timezone.utc),
                    "keywords": keywords_str,
                    "topics": topics_str,
                }
                docs.append(doc)

            return self._upload_documents(docs, "video")

        except Exception as e:
            print(f"❌ Error indexing video content: {e}")
            return False

    def index_document_content(self, document_data: Dict) -> bool:
        """
        Index document content in the unified index

        Args:
            document_data: Document data with markdown content and metadata
        """
        document_id = document_data.get("id", "unknown")
        document_name = document_data.get("name", "Unknown Document")
        markdown_content = document_data.get("content", "")

        print(f"📤 Indexing document content: {document_name} (ID: {document_id})")

        try:
            if not markdown_content:
                print("❌ No content found")
                return False

            print(f"  📊 Processing document content ({len(markdown_content)} characters)")

            # Convert document to chunks
            chunks = self._process_document_to_chunks(markdown_content)
            print(f"  📊 Created {len(chunks)} document chunks")

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
            if not texts:
                print("❌ No valid text found")
                return False

            # Generate embeddings
            print(f"  🔄 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embed_texts_batch(texts)

            # Prepare documents for unified index
            docs = []
            for chunk, embedding in zip(chunks, embeddings):
                if not embedding:
                    continue

                doc = {
                    "id": str(uuid.uuid4()),
                    "content_type": "document",
                    "source_id": document_id,
                    "source_name": document_name,
                    "text": chunk.get("text", ""),
                    "vector": embedding,
                    "chunk_index": chunk.get("chunk_index", 0),

                    # Video-specific fields (null for documents)
                    "start_seconds": None,
                    "start_time": None,
                    "end_seconds": None,

                    # Document-specific fields
                    "section_title": chunk.get("section_title", ""),
                    "document_type": document_data.get("type", "markdown"),

                    # Common metadata
                    "created_date": datetime.now(timezone.utc),
                    "keywords": "",  # Could be extracted from document metadata if available
                    "topics": "",  # Could be extracted from document metadata if available
                }
                docs.append(doc)

            return self._upload_documents(docs, "document")

        except Exception as e:
            print(f"❌ Error indexing document content: {e}")
            return False

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

    def _upload_documents(self, docs: List[Dict], content_type: str) -> bool:
        """Upload documents to the unified index"""
        if not docs:
            print("❌ No valid documents to upload")
            return False

        try:
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)

            print(f"  📤 Uploading {len(docs)} {content_type} chunks...")
            result = search_client.upload_documents(docs)
            succeeded = sum(1 for r in result if r.succeeded)

            print(f"✅ Uploaded {succeeded}/{len(docs)} {content_type} chunks successfully")
            return True

        except Exception as e:
            print(f"❌ Error uploading {content_type} documents: {e}")
            return False

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


# Convenience functions for backward compatibility

def index_video_from_data(structured_data: Dict, create_new_index: bool = True) -> bool:
    """
    Index video data in the unified index
    """
    indexer = UnifiedContentIndexer()

    # Create index
    if not indexer.create_index(create_new=create_new_index):
        return False

    # Index the video
    if not indexer.index_video_content(structured_data):
        return False

    # Show stats
    indexer.get_stats()
    return True


def index_document_from_data(document_data: Dict, create_new_index: bool = False) -> bool:
    """
    Index document data in the unified index
    """
    indexer = UnifiedContentIndexer()

    # Create index (usually don't recreate if adding documents)
    if not indexer.create_index(create_new=create_new_index):
        return False

    # Index the document
    if not indexer.index_document_content(document_data):
        return False

    # Show stats
    indexer.get_stats()
    return True


# Import functions from existing modules for parsing
def parse_video_md_to_structured_data(md_file_path: str) -> Dict:
    """Parse video MD file - imported from Index_videos.py"""
    from Index_videos import parse_md_to_structured_data
    return parse_md_to_structured_data(md_file_path)


def parse_document_md_to_data(md_file_path: str) -> Dict:
    """Parse document MD file - imported from index_docs.py"""
    from index_docs import parse_md_to_document_data
    return parse_md_to_document_data(md_file_path)


def test_unified_indexer_with_video_md(md_file_path: str, create_new_index: bool = True) -> bool:
    """
    Test the unified indexer with a video MD file
    Similar to the original Index_videos.py test function

    Args:
        md_file_path: Path to the video MD file
        create_new_index: If True (default), create new index. If False, add to existing index.
    """
    print("🧪 Testing Unified Video Indexer with MD File")
    print("=" * 50)

    if create_new_index:
        print("🔄 Mode: Creating NEW unified index (will delete existing)")
    else:
        print("➕ Mode: Adding to EXISTING unified index")

    try:
        # Parse MD file to structured data
        structured_data = parse_video_md_to_structured_data(md_file_path)

        # Test the indexer
        print("\n📤 Starting video indexing process...")
        success = index_video_from_data(structured_data, create_new_index=create_new_index)

        if success:
            print("\n✅ Successfully indexed video from MD file!")
            print("🔍 The video chunks should now be searchable in the unified index")
        else:
            print("\n❌ Failed to index video from MD file")

        return success

    except Exception as e:
        print(f"❌ Error testing video indexer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_indexer_with_document_md(md_file_path: str, create_new_index: bool = False) -> bool:
    """
    Test the unified indexer with a document MD file
    Similar to the original index_docs.py test function

    Args:
        md_file_path: Path to the document MD file
        create_new_index: If True, create new index. If False (default), add to existing index.
    """
    print("🧪 Testing Unified Document Indexer with MD File")
    print("=" * 50)

    if create_new_index:
        print("🔄 Mode: Creating NEW unified index (will delete existing)")
    else:
        print("➕ Mode: Adding to EXISTING unified index")

    try:
        # Parse MD file to document data
        document_data = parse_document_md_to_data(md_file_path)

        # Test the indexer
        print("\n📤 Starting document indexing process...")
        success = index_document_from_data(document_data, create_new_index=create_new_index)

        if success:
            print("\n✅ Successfully indexed document from MD file!")
            print("🔍 The document chunks should now be searchable in the unified index")
        else:
            print("\n❌ Failed to index document from MD file")

        return success

    except Exception as e:
        print(f"❌ Error testing document indexer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function - creates new index with video, then updates with document"""
    print("🚀 Unified Content Indexer - Videos + Documents")
    print("=" * 60)

    # # Step 1: Create new index with video content
    # print("\n📹 Step 1: Creating new unified index with video content")
    # video_md_file = "videos_md/L9_18f0d24bb7e45223abf842cdc1274de65fc7d620 - Trim.md"
    #
    # video_success = test_unified_indexer_with_video_md(video_md_file, create_new_index=True)
    #
    # if not video_success:
    #     print("\n❌ Video indexing failed - stopping")
    #     return

    # Step 2: Update existing index with document content
    print("\n📄 Step 2: Adding document content to existing unified index")
    doc_md_file = "docs_md/Ex5Sol.md"

    doc_success = test_unified_indexer_with_document_md(doc_md_file, create_new_index=False)

    if not doc_success:
        print("\n❌ Document indexing failed")
        return


if __name__ == "__main__":
    main()
