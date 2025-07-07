"""
Video Chunk Indexer - Core functionality only
Receives video data and indexes it by chunks to Azure AI Search

Core workflow:
1. Take structured video data (from video processing)
2. Chunk the transcript segments
3. Generate embeddings for chunks
4. Push chunks to Azure AI Search index
"""
###############################
##############################
##############################
########################
import uuid
import openai
import tiktoken
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


class VideoChunkIndexer:
    """
    Core video chunk indexer - minimal functionality
    """

    def __init__(self):
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)
        self.index_client = SearchIndexClient(self.search_endpoint, self.credential)
        self.index_name = "video-chunks"

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
        """Create the video chunks index

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

            # Core fields for video chunks
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="video_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="start_seconds", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="start_time", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="he.microsoft"),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="my-vector-profile",
                ),
                SimpleField(name="video_name", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True,
                            sortable=True),
            ]

            # Semantic search configuration
            semantic_config = SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")],
                    keywords_fields=[SemanticField(field_name="video_name")]
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

    def process_segments_to_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Convert transcript segments to searchable chunks"""
        chunks = []

        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            # Check if segment needs chunking
            token_count = len(self.tokenizer.encode(text))

            if token_count <= 2000:
                # Segment is fine as is
                chunks.append(segment)
            else:
                # Split large segment
                print(f"  📝 Splitting segment ({token_count} tokens)")
                text_chunks = self.chunk_by_tokens(text)

                start_seconds = segment.get("start_seconds", 0)
                duration_per_chunk = 2.0  # Default 2 seconds per chunk

                for i, chunk_text in enumerate(text_chunks):
                    chunk_start = start_seconds + (i * duration_per_chunk)

                    chunk = {
                        "text": chunk_text,
                        "start_seconds": chunk_start,
                        "start_time": segment.get("start_time", "00:00:00")
                    }
                    chunks.append(chunk)

        return chunks

    def index_video(self, structured_data: Dict) -> bool:
        """
        Main function: Index a video by chunks

        Args:
            structured_data: Video data from processing (with transcript_segments)
        """
        video_id = structured_data.get("id", "unknown")
        video_name = structured_data.get("name", "Unknown Video")

        print(f"📤 Indexing video: {video_name} (ID: {video_id})")

        try:
            # Get transcript segments
            segments = structured_data.get("transcript_segments", [])
            if not segments:
                print("❌ No transcript segments found")
                return False

            print(f"  📊 Processing {len(segments)} segments")

            # Convert to chunks
            chunks = self.process_segments_to_chunks(segments)
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
                    "video_id": video_id,
                    "start_seconds": chunk.get("start_seconds", 0),
                    "start_time": chunk.get("start_time", "00:00:00"),
                    "text": chunk.get("text", ""),
                    "vector": embedding,
                    "video_name": video_name,
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
            print(f"❌ Error indexing video: {e}")
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


def index_video_from_data(structured_data: Dict, create_new_index: bool = True) -> bool:
    """
    Simple function to index video data

    Args:
        structured_data: The output from video processing
        create_new_index: If True (default), create new index. If False, add to existing index.
    """
    indexer = VideoChunkIndexer()

    # Create index
    if not indexer.create_index(create_new=create_new_index):
        return False

    # Index the video
    if not indexer.index_video(structured_data):
        return False

    # Show stats
    indexer.get_stats()
    return True


def parse_md_to_structured_data(md_file_path: str) -> Dict:
    """
    Parse MD file and convert to structured data format expected by indexer
    """
    import re

    print(f"📖 Reading MD file: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract video information
    video_id_match = re.search(r'\*\*Video ID\*\*: (\w+)', content)
    video_id = video_id_match.group(1) if video_id_match else "unknown"

    duration_match = re.search(r'\*\*Duration\*\*: ([\d:]+)', content)
    duration = duration_match.group(1) if duration_match else "N/A"

    language_match = re.search(r'\*\*Language\*\*: ([\w-]+)', content)
    language = language_match.group(1) if language_match else "he-IL"

    speakers_match = re.search(r'\*\*Speakers\*\*: (.+)', content)
    speakers = [speakers_match.group(1)] if speakers_match else ["מרצה"]

    created_match = re.search(r'\*\*Created\*\*: (.+)', content)
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

    # Create structured data
    structured_data = {
        "id": video_id,
        "name": f"Video {video_id}",
        "duration": duration,
        "language": language,
        "speakers": speakers,
        "keywords": keywords,
        "topics": topics,
        "created_date": created_date,
        "transcript_segments": transcript_segments
    }

    print(f"✅ Parsed MD file:")
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


def test_indexer_with_md_file(md_file_path: str, create_new_index: bool = True) -> bool:
    """
    Test the video chunk indexer with an MD file

    Args:
        md_file_path: Path to the MD file
        create_new_index: If True (default), create new index. If False, add to existing index.
    """
    print("🧪 Testing Video Chunk Indexer with MD File")
    print("=" * 50)

    if create_new_index:
        print("🔄 Mode: Creating NEW index (will delete existing)")
    else:
        print("➕ Mode: Adding to EXISTING index")

    try:
        # Parse MD file to structured data
        structured_data = parse_md_to_structured_data(md_file_path)

        # Test the indexer
        print("\n📤 Starting indexing process...")
        success = index_video_from_data(structured_data, create_new_index=create_new_index)

        if success:
            print("\n✅ Successfully indexed video from MD file!")
            print("🔍 The chunks should now be searchable in the index")
        else:
            print("\n❌ Failed to index video from MD file")

        return success

    except Exception as e:
        print(f"❌ Error testing indexer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function - can test with MD file or just setup index"""
    print("🚀 Video Chunk Indexer - Core Functionality")
    print("=" * 50)

    # Test with the specific MD file
    md_file = "videos_md/L9_18f0d24bb7e45223abf842cdc1274de65fc7d620 - Trim.md"

    print("🧪 Testing with MD file...")
    success = test_indexer_with_md_file(md_file, False)

    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed")


if __name__ == "__main__":
    main()
