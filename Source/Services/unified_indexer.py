"""
Unified Content Indexer - Unifies videos and documents in one index
Receives content from different types and stores them in a shared index with flexible fields

Core workflow:
1. Take content from videos OR documents
2. Chunk the content appropriately
3. Generate embeddings for chunks
4. Push chunks to unified Azure AI Search index with flexible schema
"""
import asyncio
import uuid
import tiktoken
import re
import os
import ssl
import httpx
from typing import List, Dict
from datetime import datetime, timezone
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile,
    SearchField, SemanticConfiguration, SemanticPrioritizedFields, SemanticField, SemanticSearch
)
from openai import AsyncAzureOpenAI
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

from Config.logging_config import setup_logging

logger = setup_logging()


class UnifiedContentIndexer:
    """
    Unified indexer for different content types - videos and documents
    Uses flexible schema that fits both types
    """

    def __init__(self, openai_client=None):
        """
        Initialize unified content indexer

        Args:
            openai_client: Shared OpenAI client instance (optional)
        """
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.index_client = SearchIndexClient(self.search_endpoint, self.credential)
        self.index_name = INDEX_NAME

        # Use provided OpenAI client or create fallback instance with ownership tracking
        if openai_client is not None:
            self.openai_client = openai_client
            self._owns_client = False
        else:
            # Configure async OpenAI client as fallback
            self.openai_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2023-05-15",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self._owns_client = True

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def close(self):
        """Close the OpenAI client connection only if we own it"""
        # Close only if we created the client here
        if getattr(self, "_owns_client", False) and self.openai_client is not None:
            try:
                await self.openai_client.close()
                logger.info("Local OpenAI client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing local OpenAI client: {e}")

    def create_index(self, create_new: bool = True) -> bool:
        """Create the unified content chunks index with flexible schema

        Args:
            create_new: If True (default), delete existing index and create new one.
                       If False, use existing index or create if doesn't exist.
        """
        logger.info(f"Setting up unified index: {self.index_name}")

        try:
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                if create_new:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.index_client.delete_index(self.index_name)
                    logger.info(f"Creating new unified index: {self.index_name}")
                else:
                    logger.info(f"Using existing unified index: {self.index_name}")
                    return True
            except ResourceNotFoundError:
                logger.info(f"Creating new unified index: {self.index_name}")

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
                SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="course_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True,
                            sortable=True),

                # File name field - extracted based on content type
                SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, facetable=True),

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

                # Video-specific fields
                SimpleField(name="start_time", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="end_time", type=SearchFieldDataType.String, filterable=True, sortable=True),
                SearchableField(name="keywords", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SearchableField(name="topics", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),

                # Document-specific fields
                SearchableField(name="section_title", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
            ]

            # Semantic search configuration
            semantic_config = SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")],
                    keywords_fields=[
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
            logger.info("Unified index created successfully")

            # # Print detailed index schema
            # logger.info("\n Created index schema - all fields:")
            # logger.info("=" * 80)
            # for field in fields:
            #     field_info = f"   {field.name} ({field.type})"
            #
            #     # Add additional properties
            #     properties = []
            #     if hasattr(field, 'key') and field.key:
            #         properties.append("KEY")
            #     if hasattr(field, 'searchable') and field.searchable:
            #         properties.append("SEARCHABLE")
            #     if hasattr(field, 'filterable') and field.filterable:
            #         properties.append("FILTERABLE")
            #     if hasattr(field, 'sortable') and field.sortable:
            #         properties.append("SORTABLE")
            #     if hasattr(field, 'facetable') and field.facetable:
            #         properties.append("FACETABLE")
            #     if hasattr(field, 'analyzer_name') and field.analyzer_name:
            #         properties.append(f"ANALYZER: {field.analyzer_name}")
            #     if hasattr(field, 'vector_search_dimensions') and field.vector_search_dimensions:
            #         properties.append(f"VECTOR_DIM: {field.vector_search_dimensions}")
            #
            #     if properties:
            #         field_info += f" [{', '.join(properties)}]"
            #
            #     logger.info(field_info)
            #
            # logger.info("=" * 80)
            # logger.info(" Field explanations:")
            # logger.info("   id - unique identifier for each chunk")
            # logger.info("   content_type - content type (video/document)")
            # logger.info("   source_id - source identifier (video_id/document_id)")
            # logger.info("   text - textual content")
            # logger.info("   vector - embedding vector")
            # logger.info("   chunk_index - chunk number")
            # logger.info("   start_time - start time (video only)")
            # logger.info("   end_time - end time (video only)")
            # logger.info("   section_title - section title (documents only)")
            # logger.info("   created_date - creation date")
            # logger.info("   keywords - keywords")
            # logger.info("   topics - topics")
            # logger.info("=" * 80)

            return True

        except Exception as e:
            logger.info(f"Error creating unified index: {e}")
            return False

    async def embed_texts_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"   Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

            try:
                response = await self.openai_client.embeddings.create(
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.info(f"Error generating embeddings: {e}")
                embeddings.extend([[] for _ in batch])

        return embeddings

    def detect_sentence_endings(self, text: str) -> List[str]:
        """
        Sentence ending detection - function that identifies sentence endings
        Each sentence will be a basic chunk
        """

        # Advanced regex for identifying sentence endings in Hebrew and English
        sentence_patterns = [
            r'[.!?]+\s+',  # period/exclamation/question + space
            r'[.!?]+$',  # period/exclamation/question at end of line
            r'\n\s*\n',  # empty line (paragraph separator)
            r'[.!?]+\s*\n',  # period + new line
        ]

        sentence_regex = re.compile('|'.join(sentence_patterns), re.MULTILINE)

        sentences = []
        last_end = 0

        for match in sentence_regex.finditer(text):
            sentence = text[last_end:match.end()].strip()
            if sentence and len(sentence) > 10:  # filter sentences that are too short
                sentences.append(sentence)
            last_end = match.end()

        # Add the last part if exists
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining and len(remaining) > 10:
                sentences.append(remaining)

        return sentences

    def merge_sentences_by_length(self, sentences: List[str], max_length) -> List[Dict]:
        """
        Merge sentences into chunks by desired size
        Like Video Indexer that merges segments by time
        """

        if not sentences:
            return []

        chunks = []
        current_chunk = {
            "sentences": [],
            "text": "",
            "sentence_count": 0,
            "character_count": 0
        }

        for i, sentence in enumerate(sentences):
            # Check if adding the sentence would exceed maximum
            potential_text = current_chunk["text"] + (" " if current_chunk["text"] else "") + sentence
            potential_length = len(potential_text)

            if potential_length <= max_length or current_chunk["sentence_count"] == 0:
                # Add sentence to current chunk
                current_chunk["sentences"].append(sentence)
                current_chunk["text"] = potential_text
                current_chunk["sentence_count"] += 1
                current_chunk["character_count"] = potential_length

            else:
                # Current chunk is full - finish it and start new one
                if current_chunk["sentences"]:
                    chunk_info = {
                        "text": current_chunk["text"],
                        "sentence_count": current_chunk["sentence_count"],
                        "character_count": current_chunk["character_count"],
                        "chunk_index": len(chunks)
                    }
                    chunks.append(chunk_info)

                    # Start new chunk
                current_chunk = {
                    "sentences": [sentence],
                    "text": sentence,
                    "sentence_count": 1,
                    "character_count": len(sentence)
                }

        # Add the last chunk
        if current_chunk["sentences"]:
            chunk_info = {
                "text": current_chunk["text"],
                "sentence_count": current_chunk["sentence_count"],
                "character_count": current_chunk["character_count"],
                "chunk_index": len(chunks)
            }
            chunks.append(chunk_info)
        return chunks

    def sentence_based_chunking(self, text: str, max_chunk_length) -> List[Dict]:
        """
        Sentence-based chunking - main function
        First checks if text is natural size, only if exceeds then splits into sentences
        """

        # First check: is the text natural size?
        if len(text) <= max_chunk_length:
            return [{
                "text": text,
                "sentence_count": 1,  # considered as one sentence
                "character_count": len(text),
                "chunk_index": 0
            }]

        # Step 1: sentence identification
        sentences = self.detect_sentence_endings(text)

        if not sentences:
            return [{
                "text": text,
                "sentence_count": 1,
                "character_count": len(text),
                "chunk_index": 0
            }]

        # Step 2: merge sentences into chunks
        chunks = self.merge_sentences_by_length(sentences, max_chunk_length)

        return chunks

    def _process_video_segments_to_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Convert video transcript segments to searchable chunks"""
        chunks = []
        chunk_idx = 0

        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            # Segment is fine as is - use the calculated end_time from parsing
            end_seconds = segment.get("end_seconds", 0.0)
            end_time_formatted = convert_seconds_to_timestamp(end_seconds)

            chunk = {
                "text": text,
                "chunk_index": chunk_idx,
                "start_time": segment.get("start_time", "00:00:00"),
                "end_time": end_time_formatted
            }
            chunks.append(chunk)
            chunk_idx += 1
        return chunks

    def _process_document_to_chunks(self, markdown_content: str) -> List[Dict]:
        """Convert document markdown to searchable chunks using sentence-based chunking"""
        logger.info("Processing document with sentence-based chunking")

        # Split by headers to preserve sections
        sections = re.split(r'\n#+\s+', markdown_content)
        all_chunks = []
        global_chunk_idx = 0

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

            # Use sentence-based chunking for this section
            section_chunks = self.sentence_based_chunking(section_body, max_chunk_length=2000)

            # Add metadata to each chunk
            for chunk in section_chunks:
                chunk.update({
                    "chunk_index": global_chunk_idx,
                    "section_title": section_title
                })
                all_chunks.append(chunk)
                global_chunk_idx += 1

        logger.info(f"Created {len(all_chunks)} sentence-based chunks")
        return all_chunks

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

            logger.info(f"Unified Index Statistics:")
            logger.info(f"   Total chunks: {total_docs}")
            logger.info(f"   Video chunks: {video_count}")
            logger.info(f"   Document chunks: {doc_count}")

            return stats

        except Exception as e:
            logger.info(f"Error getting stats: {e}")
            return {}

    def delete_content_by_source(self, source_id: str, content_type: str = None) -> Dict:
        """
        Delete all content related to a specific source (video or document) from the index

        Args:
            source_id: Source identifier (video_id or document_id)
            content_type: Content type ('video' or 'document'). If None, will delete from all types

        Returns:
            Dict with deletion details
        """
        try:
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)

            # Build search filter
            if content_type:
                filter_query = f"source_id eq '{source_id}' and content_type eq '{content_type}'"
            else:
                filter_query = f"source_id eq '{source_id}'"

            logger.info(f"Searching for content to delete: {filter_query}")

            # Search all documents related to source
            results = search_client.search(
                search_text="*",
                filter=filter_query,
                select=["id", "content_type", "chunk_index"],
                include_total_count=True
            )

            # Collect all IDs for deletion
            docs_to_delete = []
            chunks_by_type = {"video": 0, "document": 0}

            for result in results:
                docs_to_delete.append({"id": result["id"]})
                chunks_by_type[result.get("content_type", "unknown")] += 1

            total_found = results.get_count()

            if not docs_to_delete:
                logger.info(f"No content found for deletion for source_id: {source_id}")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "source_id": source_id,
                    "message": "No content found for deletion"
                }

            logger.info(f"Found {total_found} chunks for deletion:")
            logger.info(f"   Video chunks: {chunks_by_type['video']}")
            logger.info(f"   Document chunks: {chunks_by_type['document']}")

            # Perform deletion
            delete_results = search_client.delete_documents(docs_to_delete)

            # Count successful deletions
            successful_deletes = sum(1 for r in delete_results if r.succeeded)
            failed_deletes = len(delete_results) - successful_deletes

            if failed_deletes > 0:
                logger.info(f"{failed_deletes} deletions failed")

            logger.info(f"Successfully deleted {successful_deletes} chunks for {source_id}")

            # Update statistics
            self.get_stats()

            return {
                "success": True,
                "deleted_count": successful_deletes,
                "failed_count": failed_deletes,
                "source_id": source_id,
                "content_type": content_type,
                "chunks_by_type": chunks_by_type,
                "message": f"Successfully deleted {successful_deletes} chunks"
            }

        except Exception as e:
            logger.info(f"Error deleting content: {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "source_id": source_id,
                "error": str(e),
                "message": f"Deletion error: {e}"
            }

    #
    # def update_content_file(self, blob_path: str, force_update: bool = False) -> Dict:
    #     """
    #     עדכון קובץ קיים באינדקס - מחיקת הגרסה הישנה והוספת הגרסה החדשה
    #
    #     Args:
    #         blob_path: נתיב הקובץ ב-blob storage
    #         force_update: אם True, יעדכן גם אם הקובץ לא קיים באינדקס
    #
    #     Returns:
    #         Dict עם פרטי העדכון
    #     """
    #     try:
    #         blob_manager = BlobManager()
    #
    #         # זיהוי סוג התוכן
    #         content_type = _detect_content_type_from_path(blob_path)
    #         logger.info(f"מעדכן קובץ: {blob_path} (סוג: {content_type})")
    #
    #         # קריאת הקובץ החדש
    #         if content_type == "video":
    #             new_data = parse_video_md_from_blob(blob_path, blob_manager)
    #             source_id = new_data.get("id", "unknown")
    #         elif content_type == "document":
    #             new_data = parse_document_md_from_blob(blob_path, blob_manager)
    #             source_id = new_data.get("id", "unknown")
    #         else:
    #             return {
    #                 "success": False,
    #                 "error": f"סוג קובץ לא נתמך: {content_type}",
    #                 "message": "עדכון נכשל - סוג קובץ לא נתמך"
    #             }
    #
    #         # בדיקה אם הקובץ קיים באינדקס
    #         search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)
    #         existing_results = search_client.search(
    #             search_text="*",
    #             filter=f"source_id eq '{source_id}' and content_type eq '{content_type}'",
    #             include_total_count=True,
    #             top=1
    #         )
    #
    #         existing_count = existing_results.get_count()
    #
    #         if existing_count == 0 and not force_update:
    #             logger.info(f"הקובץ {source_id} לא קיים באינדקס")
    #             return {
    #                 "success": False,
    #                 "source_id": source_id,
    #                 "message": "הקובץ לא קיים באינדקס. השתמש ב-force_update=True כדי להוסיף אותו"
    #             }
    #
    #         # מחיקת הגרסה הישנה (אם קיימת)
    #         if existing_count > 0:
    #             logger.info(f"מוחק גרסה ישנה של {source_id} ({existing_count} chunks)")
    #             delete_result = self.delete_content_by_source(source_id, content_type)
    #             if not delete_result["success"]:
    #                 return {
    #                     "success": False,
    #                     "source_id": source_id,
    #                     "error": delete_result.get("error", "שגיאה במחיקה"),
    #                     "message": "עדכון נכשל - לא ניתן למחוק גרסה ישנה"
    #                 }
    #
    #         # הוספת הגרסה החדשה
    #         logger.info(f"מוסיף גרסה חדשה של {source_id}")
    #         index_result = index_content_files([blob_path], create_new_index=False)
    #
    #         # בדיקה אם ההוספה הצליחה
    #         if "Indexing completed:" in index_result:
    #             # ספירת chunks חדשים
    #             new_results = search_client.search(
    #                 search_text="*",
    #                 filter=f"source_id eq '{source_id}' and content_type eq '{content_type}'",
    #                 include_total_count=True,
    #                 top=1
    #             )
    #             new_count = new_results.get_count()
    #
    #             logger.info(f"עדכון הושלם בהצלחה עבור {source_id}")
    #             logger.info(f"  Chunks חדשים: {new_count}")
    #
    #             return {
    #                 "success": True,
    #                 "source_id": source_id,
    #                 "content_type": content_type,
    #                 "old_chunks": existing_count,
    #                 "new_chunks": new_count,
    #                 "message": f"עדכון הושלם: {existing_count} → {new_count} chunks"
    #             }
    #         else:
    #             return {
    #                 "success": False,
    #                 "source_id": source_id,
    #                 "error": "שגיאה בהוספת תוכן חדש",
    #                 "message": "עדכון נכשל - לא ניתן להוסיף תוכן חדש"
    #             }
    #
    #     except Exception as e:
    #         logger.info(f"שגיאה בעדכון קובץ: {e}")
    #         return {
    #             "success": False,
    #             "source_id": source_id if 'source_id' in locals() else "unknown",
    #             "error": str(e),
    #             "message": f"עדכון נכשל: {e}"
    #         }

    def list_content_sources(self, content_type: str = None) -> Dict:
        """
        Display list of all sources in the index

        Args:
            content_type: Content type for filtering ('video' or 'document'). If None, will show all

        Returns:
            Dict with list of sources and their details
        """
        try:
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)

            # Build filter
            if content_type:
                filter_query = f"content_type eq '{content_type}'"
            else:
                filter_query = None

            # Search with grouping by source_id
            results = search_client.search(
                search_text="*",
                filter=filter_query,
                select=["source_id", "content_type"],
                facets=["source_id", "content_type"]
            )

            # Collect unique sources
            sources = {}
            for result in results:
                source_id = result.get("source_id")
                if source_id not in sources:
                    sources[source_id] = {
                        "source_id": source_id,
                        "content_type": result.get("content_type", "unknown"),
                        "chunk_count": 0
                    }
                sources[source_id]["chunk_count"] += 1

            sources_list = list(sources.values())

            logger.info(f"List of sources in index:")
            logger.info(f"   Total sources: {len(sources_list)}")

            for source in sources_list:
                logger.info(f"   {source['source_id']} ({source['content_type']}) - {source['chunk_count']} chunks")

            return {
                "success": True,
                "sources": sources_list,
                "total_sources": len(sources_list),
                "content_type_filter": content_type
            }

        except Exception as e:
            logger.info(f"Error displaying sources: {e}")
            return {
                "success": False,
                "sources": [],
                "error": str(e)
            }


def _detect_content_type_from_path(blob_path: str) -> str:
    """
    Detect content type by file path
    Returns 'video' if path contains 'Videos_md' or 'document' if contains 'Docs_md'
    """
    if "videos_md" in blob_path.lower():
        return "video"
    elif "docs_md" in blob_path.lower():
        return "document"
    else:
        # Default - try to detect by extension
        if blob_path.lower().endswith('.md'):
            return "document"  # Default for documents
        return "unknown"


async def parse_video_md_from_blob(blob_path: str, blob_manager: BlobManager) -> Dict:
    """
    Parse video MD file from blob storage and convert to structured data format expected by indexer
    """

    logger.info(f"Reading video MD file from blob: {blob_path}")

    # Download content from blob to memory
    file_bytes = await blob_manager.download_to_memory(blob_path)
    if not file_bytes:
        raise Exception(f"Failed to download blob: {blob_path}")

    content = file_bytes.decode('utf-8')

    # Extract video information - support both Hebrew and English formats
    video_id_match = re.search(r'\*\*(?:Video ID|מזהה וידאו)\*\*: (\w+)', content)
    video_id = video_id_match.group(1) if video_id_match else "unknown"

    created_match = re.search(r'\*\*(?:Created|תאריך יצירה)\*\*: (.+)', content)
    created_date = created_match.group(1) if created_match else datetime.now().isoformat()

    # Extract video duration
    duration_match = re.search(r'\*\*(?:Duration|משך זמן)\*\*: (.+)', content)
    total_duration_str = duration_match.group(1) if duration_match else "00:00:00"
    total_duration_seconds = convert_timestamp_to_seconds(total_duration_str)

    # Extract keywords
    keywords_match = re.search(r'## (?:Keywords|מילות מפתח)\n`(.+)`', content)
    keywords = []
    if keywords_match:
        keywords_text = keywords_match.group(1)
        keywords = [kw.strip() for kw in keywords_text.split('`,') if kw.strip()]
        # Clean up the last keyword
        if keywords:
            keywords[-1] = keywords[-1].rstrip('`')

    # Extract topics
    topics_match = re.search(r'## (?:Topics|נושאים)\n`(.+)`', content)
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

        # Calculate end time based on next segment or total duration
        if i + 1 < len(timestamp_matches):
            # Use start time of next segment as end time
            next_timestamp = timestamp_matches[i + 1][0]
            end_seconds = convert_timestamp_to_seconds(next_timestamp)
        else:
            # Last segment - use total video duration
            end_seconds = total_duration_seconds

        segment = {
            "text": text.strip(),
            "start_time": timestamp,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
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
        "keywords": keywords,
        "topics": topics,
        "created_date": created_date,
        "transcript_segments": transcript_segments
    }

    logger.info(f"Parsed video MD file:")
    logger.info(f"  - Video ID: {video_id}")
    logger.info(f"  - Total Duration: {total_duration_str} ({total_duration_seconds} seconds)")
    logger.info(f"  - Segments: {len(transcript_segments)}")
    logger.info(f"  - Keywords: {len(keywords)}")
    logger.info(f"  - Topics: {len(topics)}")

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


def convert_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp format like '0:00:01.03'"""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    except:
        return "0:00:00.00"


async def parse_document_md_from_blob(blob_path: str, blob_manager: BlobManager) -> Dict:
    """
    Parse document MD file from blob storage and convert to document data format expected by indexer
    """
    logger.info(f"Reading document MD file from blob: {blob_path}")

    # Download content from blob to memory
    file_bytes = await blob_manager.download_to_memory(blob_path)
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

    logger.info(f"Parsed document MD file")
    return document_data

async def index_content_files(blob_paths: List[str], create_new_index: bool = False, openai_client=None, blob_manager=None) -> str:
    """
    FULLY ASYNC: Index MD files from blob storage to unified index with batching and concurrency.
    This function is designed to run in FastAPI BackgroundTasks.

    Args:
        blob_paths: List of blob paths of MD files (e.g., ["Videos_md/video.md", "Docs_md/doc.md"])
        create_new_index: Whether to create new index (True) or add to existing index (False)
        openai_client: Shared OpenAI client instance (optional)
        blob_manager: Shared BlobManager instance (optional)

    Returns:
        Success message after completion
    """
    indexer = None
    created_local_blob_manager = False
    try:
        logger.info(f"Starting async indexing of {len(blob_paths)} files")

        indexer = UnifiedContentIndexer(openai_client=openai_client)

        # Use provided blob manager or create fallback
        if blob_manager is None:
            blob_manager = BlobManager()
            created_local_blob_manager = True

        # Create/initialize index
        if not indexer.create_index(create_new=create_new_index):
            error_msg = "Failed to create/initialize index"
            logger.error(error_msg)
            return error_msg

        # Process files in small batches with concurrency control
        batch_size = 3  # Small batches to avoid memory issues
        total_processed = 0
        total_chunks = 0
        total_errors = 0

        logger.info(f"Processing in batches of {batch_size} files")

        for i in range(0, len(blob_paths), batch_size):
            batch = blob_paths[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(blob_paths) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

            # Process batch with controlled concurrency
            batch_docs = []
            batch_results = await asyncio.gather(
                *[_process_single_file_safe(blob_path, blob_manager, indexer) for blob_path in batch],
                return_exceptions=True
            )

            # Collect results from batch
            for blob_path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {blob_path}: {result}")
                    total_errors += 1
                elif result and len(result) > 0:
                    file_docs, content_type = result
                    batch_docs.extend(file_docs)
                    total_processed += 1
                    logger.info(f"Processed {blob_path} ({content_type}): {len(file_docs)} chunks")
                else:
                    logger.warning(f"Skipped {blob_path} (empty or unsupported)")
                    total_errors += 1

            # Upload batch to index immediately (smaller uploads)
            if batch_docs:
                try:
                    search_client = SearchClient(indexer.search_endpoint, INDEX_NAME, indexer.credential)

                    logger.info(f"Uploading batch {batch_num}: {len(batch_docs)} chunks")
                    results = search_client.upload_documents(batch_docs)

                    succeeded = sum(1 for r in results if r.succeeded)
                    failed = len(results) - succeeded
                    total_chunks += succeeded

                    logger.info(f"Batch {batch_num} uploaded: {succeeded} succeeded, {failed} failed")

                    if failed > 0:
                        total_errors += failed

                except Exception as e:
                    logger.error(f"Error uploading batch {batch_num}: {e}")
                    total_errors += len(batch_docs)

            # Yield control between batches to prevent blocking
            await asyncio.sleep(0.2)

        # Final summary
        logger.info(f"Indexing completed!")
        logger.info(f"   Files processed: {total_processed}/{len(blob_paths)}")
        logger.info(f"   Total chunks indexed: {total_chunks}")
        logger.info(f"   Errors: {total_errors}")

        # Display final index statistics
        indexer.get_stats()

        return f"Indexing completed: {total_processed} files processed, {total_chunks} chunks indexed, {total_errors} errors"

    except Exception as e:
        error_msg = f"Indexing failed: {str(e)}"
        logger.error(error_msg)
        return error_msg
    finally:
        if indexer is not None:
            await indexer.close()  # safe: closes only if it owns the client
        if created_local_blob_manager and blob_manager is not None:
            await blob_manager.close()


async def _process_single_file_safe(blob_path: str, blob_manager: BlobManager, indexer: UnifiedContentIndexer):
    """
    Safely process a single file with error handling
    Returns (docs, content_type) or None if failed
    """
    try:
        return await _process_single_file(blob_path, blob_manager, indexer)
    except Exception as e:
        logger.error(f"Error processing file {blob_path}: {e}")
        return None


async def _process_single_file(blob_path: str, blob_manager: BlobManager, indexer: UnifiedContentIndexer):
    """
    Process a single file and return its documents
    Returns (docs, content_type) tuple
    """
    # Detect content type from path
    content_type = _detect_content_type_from_path(blob_path)

    if content_type == "video":
        # Parse video MD file
        video_data = await parse_video_md_from_blob(blob_path, blob_manager)
        if not video_data or not video_data.get("transcript_segments"):
            return None

        # Extract file name from video content
        file_bytes = await blob_manager.download_to_memory(blob_path)
        content = file_bytes.decode('utf-8') if file_bytes else ""
        file_name = _extract_file_name_from_video(content)

        # Convert to chunks
        chunks = indexer._process_video_segments_to_chunks(video_data["transcript_segments"])

        # Generate embeddings for chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await indexer.embed_texts_batch(texts, batch_size=8)

        # Create documents for index
        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "id": str(uuid.uuid4()),
                "content_type": "video",
                "source_id": video_data["id"],
                "course_id": _extract_course_id_from_path(blob_path),
                "file_name": file_name,
                "text": chunk["text"],
                "vector": embedding,
                "chunk_index": chunk["chunk_index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "keywords": ", ".join(video_data.get("keywords", [])),
                "topics": ", ".join(video_data.get("topics", [])),
                "created_date": datetime.now(timezone.utc).isoformat()
            }
            docs.append(doc)

        return docs, "video"

    elif content_type == "document":
        # Parse document MD file
        doc_data = await parse_document_md_from_blob(blob_path, blob_manager)
        if not doc_data or not doc_data.get("content"):
            return None

        # Extract file name from document content
        file_name = _extract_file_name_from_document(doc_data["content"])

        # Convert to chunks
        chunks = indexer._process_document_to_chunks(doc_data["content"])

        # Generate embeddings for chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await indexer.embed_texts_batch(texts, batch_size=8)

        # Create documents for index
        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "id": str(uuid.uuid4()),
                "content_type": "document",
                "source_id": doc_data["id"],
                "course_id": _extract_course_id_from_path(blob_path),
                "file_name": file_name,
                "text": chunk["text"],
                "vector": embedding,
                "chunk_index": chunk["chunk_index"],
                "section_title": chunk.get("section_title", ""),
                "created_date": datetime.now(timezone.utc).isoformat()
            }
            docs.append(doc)

        return docs, "document"

    else:
        logger.warning(f"Unsupported content type: {content_type} for {blob_path}")
        return None


def _extract_course_id_from_path(blob_path: str) -> str:
    """Extract course ID from blob path like 'CS101/Section1/Videos_md/file.md'"""
    parts = blob_path.split('/')
    return parts[0] if parts else "unknown"


def _extract_file_name_from_video(content: str) -> str:
    """
    Extract file name from video content based on video name in the metadata section

    Args:
        content: Video MD file content

    Returns:
        File name extracted from video name, or empty string if not found
    """
    # Extract video name from the metadata section
    video_name_match = re.search(r'\*\*(?:שם הוידאו|Video Name)\*\*: (.+)', content)
    if video_name_match:
        video_name = video_name_match.group(1).strip()
        return video_name

    return ""


def _extract_file_name_from_document(content: str) -> str:
    """
    Extract file name from document content based on first line header

    Args:
        content: Document MD file content

    Returns:
        File name extracted from first line header, or empty string if not found
    """
    # Check if first line starts with # (header)
    lines = content.strip().split('\n')
    if lines and lines[0].strip().startswith('#'):
        # Extract header text after # and any spaces
        header = lines[0].strip()
        # Remove # and leading/trailing spaces
        file_name = re.sub(r'^#+\s*', '', header).strip()
        return file_name

    return ""


async def main():
    """Main function - demonstrates usage with automatic type detection and new functions"""
    logger.info("Unified Content Indexer - Videos + Documents")
    logger.info("=" * 60)

    logger.info("\nCreating unified index with automatic file type detection")

    # Define blob paths to process - type will be auto-detected from path
    blob_paths = [
        "Discrete_mathematics/Section1/Docs_md/2001.md"
    ]

    result = await index_content_files(blob_paths, create_new_index=True)
    logger.info("debug")
    logger.info(f"\n{result}")

    # # בדיקת הפונקציות החדשות
    # logger.info("\n" + "=" * 60)
    # logger.info("בדיקת פונקציות מחיקה ועדכון חדשות")
    # logger.info("=" * 60)
    #
    # indexer = UnifiedContentIndexer()
    #
    # # 1. הצגת סטטיסטיקות ראשוניות
    # logger.info("\nסטטיסטיקות ראשוניות:")
    # initial_stats = indexer.get_stats()
    #
    # # 2. הצגת רשימת מקורות
    # logger.info("\nרשימת מקורות באינדקס:")
    # sources_result = indexer.list_content_sources()
    #
    # if sources_result["success"] and sources_result["sources"]:
    #     # 3. בדיקת מחיקה - נבחר מקור ראשון לבדיקה
    #     source_id = "2"
    #     content_type = "video"
    #
    #     logger.info(f"\nפרטי המקור הראשון לבדיקה:")
    #     logger.info(f"  source_id: {source_id}")
    #     logger.info(f"  content_type: {content_type}")
    #
    #     logger.info(f"\nבדיקת מחיקה עבור מקור: {source_id} (סוג: {content_type})")
    #
    #     # ביצוע מחיקה ישירות
    #     logger.info(f"  מוחק סרטון עם ID 2...")
    #
    #     # ביצוע מחיקה
    #     delete_result = indexer.delete_content_by_source(source_id, content_type)
    #     logger.info(f"  תוצאת מחיקה: {delete_result['message']}")
    #
    #     # בדיקת סטטיסטיקות אחרי מחיקה
    #     logger.info("\nסטטיסטיקות אחרי מחיקה:")
    #     after_delete_stats = indexer.get_stats()

    #     # # סיכום הבדיקה
    #     # logger.info("\nסיכום בדיקת הפונקציות החדשות:")
    #     # logger.info(f"  Chunks התחלתיים: {initial_stats.get('total_chunks', 0)}")
    #     # logger.info(f"  Chunks אחרי מחיקה: {after_delete_stats.get('total_chunks', 0)}")
    #     # logger.info(f"  Chunks סופיים: {final_stats.get('total_chunks', 0)}")
    #     #
    #     # if delete_result["success"]:
    #     #     logger.info("  פונקציית מחיקה עובדת תקין")
    #     # else:
    #     #     logger.info("  פונקציית מחיקה נכשלה")
    #
    #     # if update_result["success"]:
    #     #     logger.info("  פונקציית עדכון עובדת תקין")
    #     # else:
    #     #     logger.info("  פונקציית עדכון נכשלה")
    #
    # else:
    #     logger.info("לא נמצאו מקורות באינדקס לבדיקה")
    #     logger.info("הרץ קודם את הפונקציה index_content_files כדי להוסיף תוכן לאינדקס")


if __name__ == "__main__":
    logger.info("running")
    asyncio.run(main())

