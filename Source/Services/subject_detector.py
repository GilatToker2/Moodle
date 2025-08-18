"""
Subject Detection Service
Identifies subject name and type based on course files using LLM
"""

import os
from typing import List, Optional, Dict
from openai import AsyncAzureOpenAI
from Source.Services.blob_manager import BlobManager
import json
import asyncio
import traceback
from Config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL
)
from Config.logging_config import setup_logging

logger = setup_logging()

# Configuration constants
subject_max_doc_files = 10
subject_max_video_files = 5
subject_per_file_chars = 30000



class SubjectDetector:
    """Identifies subject name and type based on course files analysis"""

    def __init__(self, prompt_loader=None, blob_manager=None, openai_client=None):
        """
        Initialize SubjectDetector

        Args:
            prompt_loader: Shared prompt loader instance (optional)
            blob_manager: Shared BlobManager instance (optional)
            openai_client: Shared OpenAI client instance (optional)
        """
        # Use provided blob_manager or create fallback instance
        self.blob_manager = blob_manager if blob_manager is not None else BlobManager(container_name="processeddata")

        # Use provided OpenAI client or create fallback instance
        if openai_client is not None:
            self.client = openai_client
        else:
            # Create async OpenAI client as fallback
            self.client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )


        # Use provided prompt_loader or get shared instance
        self.prompt_loader = prompt_loader
        if not self.prompt_loader:
            from Source.Services.prompt_loader import get_prompt_loader
            self.prompt_loader = get_prompt_loader()

    async def detect_subject_info(self, course_path: str) -> Dict[str, str]:
        """
        Detect subject name and type from course files using LLM

        Args:
            course_path: Course path in blob storage (e.g., "CS101")

        Returns:
            Dictionary with 'name' and 'type' keys, or error values if detection failed
        """
        logger.info(f"Starting subject detection for course: {course_path}")

        # Find all MD files in course
        all_files = await self.blob_manager.list_files()

        # Filter MD files that belong to this course
        course_md_files = [
            f for f in all_files
            if f.startswith(course_path + "/") and f.endswith(".md") and
               ("/Videos_md/" in f or "/Docs_md/" in f)
        ]

        if not course_md_files:
            logger.info(f"No MD files found in course: {course_path}")
            return {"name": "לא זוהה", "type": "לא זוהה"}

        logger.info(f"Found {len(course_md_files)} MD files in course")

        # Separate and limit files by type
        video_files = [f for f in course_md_files if "/Videos_md/" in f]
        doc_files = [f for f in course_md_files if "/Docs_md/" in f]

        # Limit number of files
        selected_video_files = video_files[:subject_max_video_files]
        selected_doc_files = doc_files[:subject_max_doc_files]
        selected_files = selected_video_files + selected_doc_files

        logger.info(f"Selected {len(selected_video_files)} video files and {len(selected_doc_files)} doc files")

        # Collect all file contents
        file_contents = []
        for file_path in selected_files:
            logger.info(f"Processing file: {file_path}")

            content = await self.blob_manager.download_to_memory(file_path)
            if not content:
                logger.info(f"Cannot download file: {file_path}")
                continue

            try:
                md_content = content.decode('utf-8')
                # Limit content length
                if len(md_content) > subject_per_file_chars:
                    md_content = md_content[:subject_per_file_chars]
                    logger.info(f"Truncated file content to {subject_per_file_chars} characters")

                file_contents.append({
                    'path': file_path,
                    'content': md_content
                })
            except UnicodeDecodeError:
                logger.info(f"Error reading file: {file_path}")
                continue

        if not file_contents:
            logger.info("No readable content found")
            return {"name": "לא זוהה", "type": "לא זוהה"}

        # Analyze with LLM
        return await self._analyze_with_llm(file_contents)

    async def _analyze_with_llm(self, file_contents: List[dict]) -> Dict[str, str]:
        """
        Analyze files with LLM to determine subject name and type

        Args:
            file_contents: List of dictionaries with 'path' and 'content'

        Returns:
            Dictionary with 'name' and 'type' keys
        """
        logger.info(f"Analyzing {len(file_contents)} files with LLM")

        # Build file contents for prompt
        file_contents_text = ""
        for i, file_info in enumerate(file_contents, 1):
            file_contents_text += f"\n--- קובץ {i}: {file_info['path']} ---\n"
            file_contents_text += file_info['content']
            file_contents_text += "\n" + "=" * 50 + "\n"

        # Get prompts from prompt loader
        system_prompt = self.prompt_loader.get_prompt("subject_detection", "System")
        user_prompt = self.prompt_loader.get_prompt("subject_detection", "User", file_contents=file_contents_text)

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(f"Sending request to LLM with prompt length: {len(user_prompt)} characters")
        logger.info(f"Final prompt: {messages}")

        try:
            response = await self.client.chat.completions.create(
                model=AZURE_OPENAI_CHAT_COMPLETION_MODEL,
                messages=messages,
                max_tokens=100,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"LLM response: '{result}'")

            # Parse the response
            return self._parse_llm_response(result)

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {"name": "לא זוהה", "type": "לא זוהה"}

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response to extract subject name and type from JSON

        Args:
            response: LLM response text (should be JSON)

        Returns:
            Dictionary with 'name' and 'type' keys
        """
        result = {"name": "לא זוהה", "type": "לא זוהה"}

        try:
            # Try to parse as JSON
            json_data = json.loads(response)

            if "שם מקצוע" in json_data:
                result["name"] = json_data["שם מקצוע"]

            if "סוג מקצוע" in json_data:
                type_val = json_data["סוג מקצוע"]
                if type_val in ['מתמטי', 'הומני']:
                    result["type"] = type_val

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response}")
            # Fallback to text parsing
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if 'שם מקצוע' in line and ':' in line:
                    name = line.split(':', 1)[1].strip().strip('"')
                    if name:
                        result["name"] = name
                elif 'סוג מקצוע' in line and ':' in line:
                    type_val = line.split(':', 1)[1].strip().strip('"')
                    if type_val in ['מתמטי', 'הומני']:
                        result["type"] = type_val

        logger.info(f"Parsed result: {result}")
        return result


async def detect_subject_from_course(course_path: str) -> Dict[str, str]:
    """
    Convenient function for subject detection from course path

    Args:
        course_path: Course path in blob storage (e.g., "CS101")

    Returns:
        Dictionary with 'name' and 'type' keys
    """
    logger.info(f"Starting subject detection for course: {course_path}")
    detector = SubjectDetector()
    return await detector.detect_subject_info(course_path)


if __name__ == "__main__":

    async def test():
        logger.info("Subject detection testing")
        logger.info("=" * 50)

        course_path = "Information_systems"

        try:
            result = await detect_subject_from_course(course_path)
            logger.info(f"Subject info for course {course_path}:")
            logger.info(f"  Name: {result['name']}")
            logger.info(f"  Type: {result['type']}")
        except Exception as e:
            logger.error(f"Error in testing: {e}")
            traceback.print_exc()


    asyncio.run(test())
