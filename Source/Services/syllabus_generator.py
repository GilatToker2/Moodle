"""
Syllabus Generator - Creates academic syllabus from course summaries
Uses Azure OpenAI language model to create structured syllabus documents
"""

import os
import asyncio
import traceback
from typing import Dict, Optional
from openai import AsyncAzureOpenAI
from Config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL,
    CONTAINER_NAME
)
from Source.Services.blob_manager import BlobManager
from Config.logging_config import setup_logging

logger = setup_logging()


class SyllabusGenerator:
    """
    Syllabus generation system for academic courses
    """

    def __init__(self, prompt_loader=None, blob_manager=None, openai_client=None):
        """
        Initialize syllabus generation system

        Args:
            prompt_loader: Shared prompt loader instance (optional)
            blob_manager: Shared BlobManager instance (optional)
            openai_client: Shared OpenAI client instance (optional)
        """
        self.model_name = AZURE_OPENAI_CHAT_COMPLETION_MODEL

        # Use provided OpenAI client or create fallback instance
        if openai_client is not None:
            self.openai_client = openai_client
        else:
            # Create async OpenAI client as fallback
            self.openai_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )

        # Use provided blob_manager or create fallback instance
        self.blob_manager = blob_manager if blob_manager is not None else BlobManager()

        # Use provided prompt_loader (required)
        if prompt_loader is None:
            raise ValueError("prompt_loader is required - should be passed as shared instance")
        self.prompt_loader = prompt_loader

        logger.info(f"SyllabusGenerator initialized with model: {self.model_name}")

    def _get_syllabus_prompts(self, subject_name: str = None, subject_type: str = None,
                              course_summary_content: str = "") -> tuple:
        """Prepare system and user prompts for syllabus creation from course summary"""

        # Determine which section to use based on subject
        # Support both Hebrew and English subject types
        if subject_name and subject_type in ["מתמטי", "Mathematics", "Mathematical"]:
            system_section = "System – Math – Subject_name"
            user_section = "User – Math – Subject_name"
            system_prompt = self.prompt_loader.get_prompt("syllabus_generation", system_section,
                                                          subject_name=subject_name)
            user_prompt = self.prompt_loader.get_prompt("syllabus_generation", user_section,
                                                        course_summary_content=course_summary_content)
        elif subject_name and subject_type in ["הומני", "Humanities", "Humanistic"]:
            system_section = "System – Humanities – Subject_name"
            user_section = "User – Humanities – Subject_name"
            system_prompt = self.prompt_loader.get_prompt("syllabus_generation", system_section,
                                                          subject_name=subject_name)
            user_prompt = self.prompt_loader.get_prompt("syllabus_generation", user_section,
                                                        course_summary_content=course_summary_content)
        elif subject_type in ["מתמטי", "Mathematics", "Mathematical"]:
            system_section = "System – Math – General"
            user_section = "User – Math – General"
            system_prompt = self.prompt_loader.get_prompt("syllabus_generation", system_section)
            user_prompt = self.prompt_loader.get_prompt("syllabus_generation", user_section,
                                                        course_summary_content=course_summary_content)
        elif subject_type in ["הומני", "Humanities", "Humanistic"]:
            system_section = "System – Humanities – General"
            user_section = "User – Humanities – General"
            system_prompt = self.prompt_loader.get_prompt("syllabus_generation", system_section)
            user_prompt = self.prompt_loader.get_prompt("syllabus_generation", user_section,
                                                        course_summary_content=course_summary_content)
        else:
            system_section = "System – General"
            user_section = "User – General"
            system_prompt = self.prompt_loader.get_prompt("syllabus_generation", system_section)
            user_prompt = self.prompt_loader.get_prompt("syllabus_generation", user_section,
                                                        course_summary_content=course_summary_content)

        return system_prompt, user_prompt

    async def create_syllabus_from_course_summary(self, full_blob_path: str, subject_name: str = None,
                                                  subject_type: str = None) -> str | None:
        """
        Create syllabus from course summary file in blob storage
        Args:
            full_blob_path: Path to course summary file (e.g., "Intro_to_medieval_history/course_summary.md")
            subject_name: Subject name for context
            subject_type: Subject type for context
        Returns:
            Syllabus path in blob or None if failed
        """

        try:
            # Parse path to extract course_id
            # Example: "Intro_to_medieval_history/course_summary.md" -> "Intro_to_medieval_history"
            if not full_blob_path.endswith('/course_summary.md'):
                logger.info(f"Invalid path: {full_blob_path}. Should end with '/course_summary.md'")
                return None

            course_id = full_blob_path.replace('/course_summary.md', '')
            logger.info(f"CourseID: {course_id}")
            logger.info(f"Course summary path: {full_blob_path}")

            # Download course summary file
            logger.info(f"Downloading course summary file: {full_blob_path}")

            try:
                # Download file directly to memory
                file_bytes = await self.blob_manager.download_to_memory(full_blob_path)

                if not file_bytes:
                    logger.info(f"Failed to download course summary file: {full_blob_path}")
                    return None

                # Convert to text
                course_summary_content = file_bytes.decode('utf-8')

                if not course_summary_content.strip():
                    logger.info(f"Course summary file is empty: {full_blob_path}")
                    return None

                logger.info(f"Course summary loaded successfully: {len(course_summary_content)} characters")

            except Exception as e:
                logger.info(f"Error downloading course summary file {full_blob_path}: {e}")
                return None

            # Create syllabus
            logger.info(f"Creating syllabus from course summary...")

            # Prepare special prompts for syllabus creation
            system_prompt, user_prompt = self._get_syllabus_prompts(subject_name, subject_type, course_summary_content)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            logger.info(f"Final prompt: {messages}")

            # Call language model
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            syllabus_content = response.choices[0].message.content

            logger.info(f"Syllabus created successfully!")
            logger.info(f"Syllabus length: {len(syllabus_content)} characters")

            # Add title to syllabus
            if subject_name:
                syllabus_title = f"# סילבוס קורס {subject_name}\n\n"
            else:
                syllabus_title = f"# סילבוס קורס\n\n"
            syllabus_with_title = syllabus_title + syllabus_content

            # Save syllabus to blob in structure: CourseID/syllabus.md
            syllabus_blob_path = f"{course_id}/syllabus.md"

            logger.info(f"Saving syllabus to blob: {syllabus_blob_path}")

            success = await self.blob_manager.upload_text_to_blob(
                text_content=syllabus_with_title,
                blob_name=syllabus_blob_path
            )

            if success:
                logger.info(f"Syllabus saved to blob: {syllabus_blob_path}")
                return syllabus_blob_path
            else:
                logger.info(f"Failed to save syllabus to blob")
                return None

        except Exception as e:
            logger.info(f"Error in syllabus creation: {str(e)}")
            return None


async def main():
    """Main function for testing syllabus generation"""
    logger.info("Syllabus Generator - Testing")
    logger.info("=" * 50)

    generator = SyllabusGenerator()

    # Test parameters
    subject_name = "מבוא לימי הביניים"
    subject_type = "הומני"
    course_summary_path = "Intro_to_medieval_history/course_summary.md"

    logger.info(f"Testing syllabus generation:")
    logger.info(f"Course summary path: {course_summary_path}")
    logger.info(f"Subject: {subject_name} ({subject_type})")

    try:
        result = await generator.create_syllabus_from_course_summary(
            full_blob_path=course_summary_path,
            subject_name=subject_name,
            subject_type=subject_type
        )

        if result:
            logger.info(f"Syllabus created successfully!")
            logger.info(f"Syllabus saved to: {result}")
        else:
            logger.info(f"Failed to create syllabus")
            logger.info(f"Make sure the course summary exists at: {course_summary_path}")

    except Exception as e:
        logger.info(f"Error during syllabus generation: {str(e)}")
        traceback.print_exc()

    logger.info("=" * 50)
    logger.info("Testing completed")


if __name__ == "__main__":
    asyncio.run(main())
