"""
Configuration file for Azure Search Pipeline
Reads configuration from environment variables for security
"""

import os
from dotenv import load_dotenv
from Config.logging_config import setup_logging

logger = setup_logging()

# Load environment variables from .env file
load_dotenv()

# Azure Storage Configuration
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "processeddata")  # Default value if not set

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
AZURE_OPENAI_SUMMARIZE_MODEL = os.getenv("AZURE_OPENAI_SUMMARIZE_MODEL", "gpt-4o")
AZURE_OPENAI_CHAT_COMPLETION_MODEL = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL", "gpt-4.1")

# Video Indexer Configuration
VIDEO_INDEXER_ACCOUNT_ID = os.getenv("VIDEO_INDEXER_ACCOUNT_ID")
VIDEO_INDEXER_LOCATION = os.getenv("VIDEO_INDEXER_LOCATION", "westeurope")
VIDEO_INDEXER_SUB_ID = os.getenv("VIDEO_INDEXER_SUB_ID")
VIDEO_INDEXER_RG = os.getenv("VIDEO_INDEXER_RG", "Moodle-RG")
VIDEO_INDEXER_VI_ACC = os.getenv("VIDEO_INDEXER_VI_ACC", "moodle-VI")
VIDEO_INDEXER_TOKEN = os.getenv("VIDEO_INDEXER_TOKEN")

# Webhook Configuration
LOGIC_APP_URL = os.getenv("LOGIC_APP_URL")

# Azure Search Configuration
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

INDEX_NAME = os.getenv("INDEX_NAME", "moodle-index-1")

# Azure Form Recognizer Configuration
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")

# Validation - ensure critical environment variables are set
def validate_config():
    """Validate that all required environment variables are set"""
    required_vars = [
        "STORAGE_CONNECTION_STRING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "VIDEO_INDEXER_ACCOUNT_ID",
        "VIDEO_INDEXER_SUB_ID",
        "SEARCH_SERVICE_NAME",
        "SEARCH_API_KEY"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    logger.info("All required environment variables are set")

# Optional: Run validation when module is imported
# Uncomment the line below if you want automatic validation
validate_config()