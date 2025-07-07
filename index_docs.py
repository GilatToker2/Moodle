"""
Document Chunk Indexer - Core functionality for documents
Receives document markdown data and indexes it by chunks to Azure AI Search

Core workflow:
1. Take document markdown content
2. Chunk the document using text splitting
3. Generate embeddings for chunks
4. Push chunks to Azure AI Search index

Based on Index_videos.py but adapted for documents instead of videos
"""

import uuid
import openai
import tiktoken
import re
from typing import List, Dict
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


class DocumentChunkIndexer:
    """
    Core document chunk indexer - minimal functionality adapted from VideoChunkIndexer
    """

    def __init__(self):
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.index_client = SearchIndexClient(self.search_endpoint, self.credential)
        self.index_name = "document-chunks"

        # Configure OpenAI client (new version)
        from openai import AzureOpenAI
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def create_index(self, create_new: bool = True) -> bool:
        """Create the document chunks index

        Args:
            create_new: If True (default), delete existing index and create new one.
                       If False, use existing index or create if doesn't exist.
        """
        print(f"🔧 Setting up index: {self.index_name}")

        try:
            # Check if index exists
            index_exists = False
            try:
                existing_index = self.index_client.get_index(self.index_name)
                index_exists = True

                if create_new:
                    print(f"🗑️ Deleting existing index: {self.index_name}")
                    self.index_client.delete_index(self.index_name)
                    print(f"📝 Creating new index: {self.index_name}")
                else:
                    print(f"✅ Using existing index: {self.index_name}")
                    return True

            except ResourceNotFoundError:
                print(f"📝 Creating new index: {self.index_name}")

            # Vector search configuration (matching working pattern)
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

            # Core fields for document chunks
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="my-vector-profile",
                ),
                SimpleField(name="document_name", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="section_title", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True,
                            sortable=True),
            ]

            # Semantic search configuration
            semantic_config = SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")],
                    keywords_fields=[SemanticField(field_name="document_name"), SemanticField(field_name="section_title")]
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
            print("✅ Index created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating index: {e}")
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

    def chunk_by_tokens(self, text: str, max_tokens: int = 500) -> List[str]:
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

    def process_document_to_chunks(self, markdown_content: str) -> List[Dict]:
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
                print(f"  📝 Splitting section ({token_count} tokens)")
                text_chunks = self.chunk_by_tokens(section_body)

                for chunk_text in text_chunks:
                    chunks.append({
                        "text": chunk_text,
                        "chunk_index": chunk_idx,
                        "section_title": section_title
                    })
                    chunk_idx += 1

        return chunks

    def index_document(self, document_data: Dict) -> bool:
        """
        Main function: Index a document by chunks

        Args:
            document_data: Document data with markdown content and metadata
        """
        document_id = document_data.get("id", "unknown")
        document_name = document_data.get("name", "Unknown Document")
        markdown_content = document_data.get("content", "")

        print(f"📤 Indexing document: {document_name} (ID: {document_id})")

        try:
            if not markdown_content:
                print("❌ No content found")
                return False

            print(f"  📊 Processing document content ({len(markdown_content)} characters)")

            # Convert to chunks
            chunks = self.process_document_to_chunks(markdown_content)
            print(f"  📊 Created {len(chunks)} chunks")

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
            if not texts:
                print("❌ No valid text found")
                return False

            # Generate embeddings
            print(f"  🔄 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embed_texts_batch(texts)

            # Prepare documents
            docs = []
            for chunk, embedding in zip(chunks, embeddings):
                if not embedding:
                    continue

                doc = {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
                    "vector": embedding,
                    "document_name": document_name,
                    "document_type": document_data.get("type", "unknown"),
                    "section_title": chunk.get("section_title", ""),
                    "created_date": datetime.now(timezone.utc)
                }
                docs.append(doc)

            if not docs:
                print("❌ No valid documents to upload")
                return False

            # Upload to index
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)

            print(f"  📤 Uploading {len(docs)} chunks...")
            result = search_client.upload_documents(docs)
            succeeded = sum(1 for r in result if r.succeeded)

            print(f"✅ Uploaded {succeeded}/{len(docs)} chunks successfully")
            return True

        except Exception as e:
            print(f"❌ Error indexing document: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get basic index statistics"""
        try:
            search_client = SearchClient(self.search_endpoint, self.index_name, self.credential)
            results = search_client.search("*", include_total_count=True, top=0)
            total_docs = results.get_count()

            print(f"📊 Index has {total_docs} chunks")
            return {"total_chunks": total_docs}

        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}


def index_document_from_data(document_data: Dict, create_new_index: bool = True) -> bool:
    """
    Simple function to index document data

    Args:
        document_data: The document data with content
        create_new_index: If True (default), create new index. If False, add to existing index.
    """
    indexer = DocumentChunkIndexer()

    # Create index
    if not indexer.create_index(create_new=create_new_index):
        return False

    # Index the document
    if not indexer.index_document(document_data):
        return False

    # Show stats
    indexer.get_stats()
    return True


def parse_md_to_document_data(md_file_path: str) -> Dict:
    """
    Parse MD file and convert to document data format expected by indexer
    """
    import os

    print(f"📖 Reading MD file: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract document info from filename
    base_name = os.path.splitext(os.path.basename(md_file_path))[0]

    # Create document data
    document_data = {
        "id": base_name,
        "name": base_name.replace('_', ' ').replace('-', ' '),
        "content": content,
        "type": "markdown"
    }

    print(f"✅ Parsed MD file:")
    print(f"  - Document ID: {document_data['id']}")
    print(f"  - Document Name: {document_data['name']}")
    print(f"  - Content Length: {len(content)} characters")

    return document_data


def test_indexer_with_md_file(md_file_path: str, create_new_index: bool = False) -> bool:
    """
    Test the document chunk indexer with an MD file

    Args:
        md_file_path: Path to the MD file
        create_new_index: If True (default), create new index. If False, add to existing index.
    """
    print("🧪 Testing Document Chunk Indexer with MD File")
    print("=" * 50)

    if create_new_index:
        print("🔄 Mode: Creating NEW index (will delete existing)")
    else:
        print("➕ Mode: Adding to EXISTING index")

    try:
        # Parse MD file to document data
        document_data = parse_md_to_document_data(md_file_path)

        # Test the indexer
        print("\n📤 Starting indexing process...")
        success = index_document_from_data(document_data, create_new_index=create_new_index)

        if success:
            print("\n✅ Successfully indexed document from MD file!")
            print("🔍 The chunks should now be searchable in the index")
        else:
            print("\n❌ Failed to index document from MD file")

        return success

    except Exception as e:
        print(f"❌ Error testing indexer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function - can test with MD file or just setup index"""
    print("🚀 Document Chunk Indexer - Core Functionality")
    print("=" * 50)

    # Test with a document from docs_md folder
    # md_file = "docs_md/Ex5Sol.md"
    # md_file = "docs_md/דף נוסחאות בדידה.md"
    md_file = "docs_md/bdida_tirgul_02.md"

    print("🧪 Testing with MD file...")
    success = test_indexer_with_md_file(md_file, create_new_index=False)

    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed")


if __name__ == "__main__":
    main()