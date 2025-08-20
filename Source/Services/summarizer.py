"""
Content Summarizer - Summarization system for videos and documents
Uses Azure OpenAI language model to create customized summaries
"""

import os
import asyncio
import traceback
from typing import Dict, Optional, List
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


class ContentSummarizer:
    """
    Content summarization system - videos and documents
    """

    def __init__(self, prompt_loader=None, blob_manager=None, openai_client=None):
        """
        Initialize summarization system

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

        # Use provided prompt_loader or get shared instance
        self.prompt_loader = prompt_loader
        if not self.prompt_loader:
            from Source.Services.prompt_loader import get_prompt_loader
            self.prompt_loader = get_prompt_loader()

        logger.info(f"ContentSummarizer initialized with model: {self.model_name}")


    def build_base_prompt(self, subject_name: Optional[str], subject_type: Optional[str],
                          input_type: str = "file") -> str:
        """
        Returns a Hebrew, learning-oriented summarization prompt tailored by subject_type and input_type.

        Args:
            subject_name: Name of the subject
            subject_type: "מתמטי" | "הומני" | None
            input_type: "video" | "file" (generic course file)

        Returns:
            Formatted prompt string
        """


        # Determine which section to use based on subject and input type
        if subject_name and subject_type == "מתמטי":
            section = f"System - מתמטי עם שם מקצוע - {'וידאו' if input_type == 'video' else 'קובץ'}"
            logger.info(f"DEBUG: Looking for section: {section}")
            prompt = self.prompt_loader.get_prompt("file_summarization", section,
                                          subject_name=subject_name)
            logger.info(f"DEBUG: Got prompt length: {len(prompt)}")
            return prompt
        elif subject_name and subject_type == "הומני":
            section = f"System - הומני עם שם מקצוע - {'וידאו' if input_type == 'video' else 'קובץ'}"
            return self.prompt_loader.get_prompt("file_summarization", section,
                                            subject_name=subject_name)
        elif subject_type == "מתמטי":
            section = f"System - מתמטי כללי - {'וידאו' if input_type == 'video' else 'קובץ'}"
            return self.prompt_loader.get_prompt("file_summarization", section)
        elif subject_type == "הומני":
            section = f"System - הומני כללי - {'וידאו' if input_type == 'video' else 'קובץ'}"
            return self.prompt_loader.get_prompt("file_summarization", section)
        else:
            section = f"System - כללי - {'וידאו' if input_type == 'video' else 'קובץ'}"
            return self.prompt_loader.get_prompt("file_summarization", section)

    def _get_section_summary_prompt(self, subject_name: str = None, subject_type: str = None,
                                    previous_summary: str = None) -> str:
        """Prepare prompt for complete Section summarization"""


        # Check if we have actual previous summary content
        has_previous_summary = previous_summary and previous_summary.strip()

        # Build section name based on subject and whether we have previous summary
        if has_previous_summary:
            # With previous summary versions
            if subject_name and subject_type == "מתמטי":
                section = "System - מתמטי עם שם מקצוע - עם סיכום קודם"
            elif subject_name and subject_type == "הומני":
                section = "System - הומני עם שם מקצוע - עם סיכום קודם"
            elif subject_type == "מתמטי":
                section = "System - מתמטי כללי - עם סיכום קודם"
            elif subject_type == "הומני":
                section = "System - הומני כללי - עם סיכום קודם"
            else:
                section = "System - כללי - עם סיכום קודם"

            return self.prompt_loader.get_prompt("section_summarization", section,
                                                 subject_name=subject_name,
                                                 previous_summary=previous_summary)
        else:
            # Without previous summary versions
            if subject_name and subject_type == "מתמטי":
                section = "System - מתמטי עם שם מקצוע"
            elif subject_name and subject_type == "הומני":
                section = "System - הומני עם שם מקצוע"
            elif subject_type == "מתמטי":
                section = "System - מתמטי כללי"
            elif subject_type == "הומני":
                section = "System - הומני כללי"
            else:
                section = "System - כללי"

            return self.prompt_loader.get_prompt("section_summarization", section,
                                                 subject_name=subject_name)

    def _get_course_summary_prompt(self, subject_name: str = None, subject_type: str = None) -> str:
        """Prepare prompt for reorganizing complete course content"""

        # Determine which section to use based on subject and input type
        if subject_name and subject_type == "מתמטי":
            section = "System - מתמטי עם שם מקצוע"
            prompt = self.prompt_loader.get_prompt("course_summarization", section,
                                              subject_name=subject_name)
        elif subject_name and subject_type == "הומני":
            section = "System - הומני עם שם מקצוע"
            prompt = self.prompt_loader.get_prompt("course_summarization", section,
                                              subject_name=subject_name)
        elif subject_type == "מתמטי":
            section = "System - מתמטי כללי"
            prompt = self.prompt_loader.get_prompt("course_summarization", section)
        elif subject_type == "הומני":
            section = "System - הומני כללי"
            prompt = self.prompt_loader.get_prompt("course_summarization", section)
        else:
            section = "System - כללי"
            prompt = self.prompt_loader.get_prompt("course_summarization", section)

        return prompt

    async def parse_video_md_file_from_blob(self, blob_path: str) -> Dict:
        """
        Parse video.md file from blob storage to extract transcript

        Args:
            blob_path: Path to video.md file in blob storage

        Returns:
            Dictionary with transcript content
        """
        logger.info(f"Parsing video MD file from blob: {blob_path}")

        file_bytes = await self.blob_manager.download_to_memory(blob_path)
        if not file_bytes:
            raise FileNotFoundError(f"File not found in blob: {blob_path}")

        content = file_bytes.decode('utf-8')
        full_transcript = None

        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped in ["## Full Transcript", "## טרנסקריפט מלא", "## 📄 טרנסקריפט מלא"]:
                current_section = "full_transcript"
                section_content = []
            elif line_stripped.startswith("## ") and current_section == "full_transcript":
                if section_content:
                    full_transcript = '\n'.join(section_content).strip()
                break
            else:
                if current_section:
                    section_content.append(line)

        if current_section and section_content:
            full_transcript = '\n'.join(section_content).strip()

        return {
            "full_transcript": full_transcript,
            "original_content": content
        }

    async def summarize_content(self, content: str, content_type: str = "document", subject_name: str = None,
                                subject_type: str = None, existing_summary: str = None) -> str:
        """
        Create summary for content

        Args:
            content: Content to summarize (MD text)
            content_type: Content type - "video" or "document"
            subject_name: Subject name
            subject_type: Subject type (for video only)
            existing_summary: Existing summary (for video only)

        Returns:
            Generated summary
        """
        logger.info(f"Creating summary for {content_type} content...")
        logger.info(f"Content length: {len(content)} characters")

        try:
            # Use the new build_base_prompt function
            logger.info(f"Subject name: {subject_name}")
            logger.info(f"Subject type: {subject_type}")

            if content_type.lower() == "video":
                system_prompt = self.build_base_prompt(
                    subject_name=subject_name,
                    subject_type=subject_type,
                    input_type="video"
                )
            else:
                system_prompt = self.build_base_prompt(
                    subject_name=subject_name,
                    subject_type=subject_type,
                    input_type="file"
                )

            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            logger.info(f"Final prompt: {messages}")

            # Call language model
            logger.info(f"Calling {self.model_name} for summarization...")
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Stability in summarization
                top_p=0.7
            )

            summary = response.choices[0].message.content

            logger.info(f"Summary created successfully!")
            logger.info(f"Summary length: {len(summary)} characters")

            return summary

        except Exception as e:
            logger.info(f"Error creating summary: {e}")
            return f"Error creating summary: {str(e)}"

    def _detect_content_type_from_path(self, blob_path: str) -> str:
        """
        Identify content type by file path
        Returns 'video' if path contains 'Videos_md' or 'document' if contains 'Docs_md'
        """
        logger.info(f"blob_path: {blob_path}")
        if "videos_md" in blob_path.lower():
            return "video"
        elif "docs_md" in blob_path.lower():
            return "document"
        else:
            # Default - try to identify by extension
            if blob_path.lower().endswith('.md'):
                return "document"  # Default for documents
            return "unknown"

    def _extract_section_from_path(self, blob_path: str) -> str:
        """
        Extract section name from blob path
        Example: "Section1/Processed-data/Videos_md/file.md" -> "Section1"
        """
        path_parts = blob_path.split('/')
        for part in path_parts:
            if part.lower().startswith('section'):
                return part
        return "general"  # Default if no section found

    async def summarize_md_file(self, blob_path: str, subject_name: str = None, subject_type: str = None) -> str | None:
        """
        Summarize MD file from blob with automatic content type detection and save to blob

        Args:
            blob_path: Path to MD file in blob
            subject_name: Subject name for context
            subject_type: Subject type for context

        Returns:
            Summary path in blob or None if failed
        """
        logger.info(f"Processing MD file from blob: {blob_path}")

        try:
            # Identify content type from path
            content_type = self._detect_content_type_from_path(blob_path)
            logger.info(f"Identified as type: {content_type}")

            if content_type == "unknown":
                logger.info(f"Cannot identify file type for: {blob_path}")
                return None

            # Download file directly to memory from blob
            file_bytes = await self.blob_manager.download_to_memory(blob_path)
            if not file_bytes:
                logger.info(f"Failed to download file from blob: {blob_path}")
                return None

            # Convert to text
            content = file_bytes.decode('utf-8')
            temp_file_path = None  # No temp file needed

            try:
                # If it's a video file - use advanced parsing
                if content_type == "video":
                    logger.info("Video file detected - using enhanced parsing and summarization")

                    # Parse file content directly from blob
                    parsed_data = await self.parse_video_md_file_from_blob(blob_path)

                    # Check that transcript exists
                    if not parsed_data.get("full_transcript"):
                        logger.info(f"No transcript found in video file")
                        return None

                    # Create summary with parsed parameters
                    summary = await self.summarize_content(
                        content=parsed_data["full_transcript"],
                        content_type="video",
                        subject_name=subject_name,
                        subject_type=subject_type,
                        existing_summary=parsed_data.get("existing_summary")
                    )

                # If it's a regular document - standard handling
                else:
                    logger.info("Document file - using standard processing")

                    if not content.strip():
                        logger.info(f"File is empty")
                        return None

                    # Create summary
                    summary = await self.summarize_content(
                        content=content,
                        content_type=content_type,
                        subject_name=subject_name,
                        subject_type=subject_type
                    )

                # Check that summary was created successfully
                if not summary or summary.startswith("Error"):
                    logger.info(f"Failed to create summary")
                    return None

                # Save summary to blob
                blob_summary_path = await self._save_summary_to_blob(summary, blob_path)
                if blob_summary_path:
                    logger.info(f"Summary saved to blob: {blob_summary_path}")
                    return blob_summary_path
                else:
                    logger.info(f"Failed to save summary to blob")
                    return None

            finally:
                # Delete temporary file (only if it was created)
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            logger.info(f"Error processing file: {str(e)}")
            return None

    async def _save_summary_to_blob(self, summary: str, original_blob_path: str) -> str:
        """
        Save summary to blob in structure CourseID/SectionID/file_summaries/FileID.md

        Args:
            summary: Summary to save
            original_blob_path: Original file path in blob (e.g., "CS101/Section1/Docs_md/1.md")

        Returns:
            Summary path in blob or None if failed
        """
        try:
            # Parse original path
            # Example: "CS101/Section1/Docs_md/1.md" -> ["CS101", "Section1", "Docs_md", "1.md"]
            path_parts = original_blob_path.split('/')

            if len(path_parts) < 4:
                logger.info(f"Invalid path: {original_blob_path}")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] is Docs_md or Videos_md
            filename = path_parts[3]  # 1.md

            # Extract filename without extension
            base_name = os.path.splitext(filename)[0]  # 1

            # Add title to summary
            title = f"# סיכום קובץ {base_name}\n\n"
            summary_with_title = title + summary

            # Create new summary path
            summary_blob_path = f"{course_id}/{section_id}/file_summaries/{base_name}.md"

            logger.info(f"Saving summary to blob: {summary_blob_path}")

            # Save to blob
            success = await self.blob_manager.upload_text_to_blob(
                text_content=summary_with_title,
                blob_name=summary_blob_path,
                container=CONTAINER_NAME
            )

            if success:
                return summary_blob_path
            else:
                logger.info(f"Failed to save summary to blob")
                return None

        except Exception as e:
            logger.info(f"Error saving summary to blob: {str(e)}")
            return None

    async def summarize_md_files(self, blob_paths: List[str], subject_name: str = None, subject_type: str = None) -> Dict[str, str]:
        """
        Summarize multiple MD files from blob storage with asynchronous processing and queue management
        Similar to insert_to_index function but for summarization

        Args:
            blob_paths: List of paths to MD files in blob storage
            subject_name: Subject name for context
            subject_type: Subject type for context

        Returns:
            Dictionary mapping original blob paths to summary paths (or None if failed)
        """
        logger.info(f"Starting batch summarization of {len(blob_paths)} files")
        logger.info(f"Subject: {subject_name} ({subject_type})")

        # Validate input
        if not blob_paths:
            logger.warning("Empty blob paths list provided")
            return {}

        # Check all files are MD
        for blob_path in blob_paths:
            if not blob_path.lower().endswith('.md'):
                logger.warning(f"Skipping non-MD file: {blob_path}")

        # Filter only MD files
        md_files = [path for path in blob_paths if path.lower().endswith('.md')]
        if not md_files:
            logger.error("No MD files found in the provided list")
            return {}

        logger.info(f"Processing {len(md_files)} MD files")

        # Process files in small batches with concurrency control
        batch_size = 3  # Small batches to avoid overwhelming the system
        results = {}
        total_processed = 0
        total_successful = 0
        total_failed = 0

        logger.info(f"Processing in batches of {batch_size} files")

        for i in range(0, len(md_files), batch_size):
            batch = md_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(md_files) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

            # Process batch with controlled concurrency
            batch_results = await asyncio.gather(
                *[self._process_single_file_safe(blob_path, subject_name, subject_type) for blob_path in batch],
                return_exceptions=True
            )

            # Collect results from batch
            for blob_path, result in zip(batch, batch_results):
                total_processed += 1
                if isinstance(result, Exception):
                    logger.error(f"Error processing {blob_path}: {result}")
                    results[blob_path] = None
                    total_failed += 1
                elif result:
                    logger.info(f"Successfully processed {blob_path} -> {result}")
                    results[blob_path] = result
                    total_successful += 1
                else:
                    logger.warning(f"Failed to process {blob_path}")
                    results[blob_path] = None
                    total_failed += 1

            # Small delay between batches to prevent overwhelming the system
            if i + batch_size < len(md_files):
                logger.info(f"Waiting 2 seconds before next batch...")
                await asyncio.sleep(2)

        # Final summary
        logger.info(f"Batch summarization completed!")
        logger.info(f"   Files processed: {total_processed}")
        logger.info(f"   Successful: {total_successful}")
        logger.info(f"   Failed: {total_failed}")
        logger.info(f"   Success rate: {(total_successful/total_processed*100):.1f}%" if total_processed > 0 else "   Success rate: 0%")

        return results

    async def _process_single_file_safe(self, blob_path: str, subject_name: str = None, subject_type: str = None) -> str | None:
        """
        Safely process a single file with error handling
        Returns summary path or None if failed
        """
        try:
            return await self.summarize_md_file(blob_path, subject_name, subject_type)
        except Exception as e:
            logger.error(f"Error processing file {blob_path}: {e}")
            return None

    async def summarize_section_from_blob(self, full_blob_path: str, subject_name: str = None,
                                          subject_type: str = None, previous_summary_path: str = None) -> str | None:
        """
        Summarize complete section from all summary files in blob storage
        Args:
            full_blob_path: Path to file_summaries folder (e.g., "CS101/Section1/file_summaries")
            subject_name: Subject name for context
            subject_type: Subject type for context
        Returns:
            Summary path in blob or None if failed
        """

        try:
            # Parse path: "CS101/Section1/file_summaries" -> ["CS101", "Section1", "file_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 3:
                logger.info(f"Invalid path: {full_blob_path}. Should be in format: CourseID/SectionID/file_summaries")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] should be file_summaries

            logger.info(f"CourseID: {course_id}")
            logger.info(f"SectionID: {section_id}")
            logger.info(f"file_summaries path: {full_blob_path}")

            # Use the shared blob manager instance
            all_files = await self.blob_manager.list_files()

            # Filter files in specific path
            section_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not section_files:
                logger.info(f"No summary files found in {full_blob_path}")
                return None

            logger.info(f"Found {len(section_files)} summary files in {full_blob_path}:")
            for file in section_files:
                logger.info(f"  - {file}")

            # Download and read all files directly to memory
            all_content = ""
            successful_files = []

            for file_path in section_files:
                logger.info(f"\n Downloading file to memory: {file_path}")

                try:
                    # Download file directly to memory
                    file_bytes = await self.blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # Convert to text
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"קובץ: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f" File read successfully: {len(file_content)} characters")
                        else:
                            logger.info(f"Empty file: {file_path}")
                    else:
                        logger.info(f"Failed to download file: {file_path}")

                except Exception as e:
                    logger.info(f"Error processing file {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f"Could not read any files from {full_blob_path}")
                return None

            logger.info(f"\n Total working with {len(successful_files)} files")
            logger.info(f"Total content length: {len(all_content)} characters")

            # Handle previous summary path if provided
            previous_summary = None
            if previous_summary_path:
                try:
                    logger.info(f"Trying to read previous section summary: {previous_summary_path}")
                    previous_file_bytes = await self.blob_manager.download_to_memory(previous_summary_path)
                    if previous_file_bytes:
                        previous_summary = previous_file_bytes.decode('utf-8')
                        logger.info(f"Successfully loaded previous summary: {len(previous_summary)} characters")
                    else:
                        logger.warning(f"Could not download previous summary from: {previous_summary_path}")
                except Exception as e:
                    logger.warning(f"Error loading previous summary from {previous_summary_path}: {e}")

            # Create summary
            logger.info(f"\n Creating section summary...")

            # Prepare special prompt for section summary
            system_prompt = self._get_section_summary_prompt(subject_name, subject_type, previous_summary)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": all_content
                }
            ]

            # Call language model
            logger.info(f"Final prompt: {messages}")
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            section_summary = response.choices[0].message.content

            logger.info(f" Section summary created successfully!")
            logger.info(f" Summary length: {len(section_summary)} characters")

            # Add title to section summary
            section_title = f"# סיכום פרק {section_id}\n\n"
            section_summary_with_title = section_title + section_summary

            # Save summary to blob in new structure: CourseID/section_summaries/SectionID.md
            summary_blob_path = f"{course_id}/section_summaries/{section_id}.md"

            logger.info(f"Saving section summary to blob: {summary_blob_path}")

            success = await self.blob_manager.upload_text_to_blob(
                text_content=section_summary_with_title,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f"Section summary saved to blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f" Failed to save section summary to blob")
                return None

        except Exception as e:
            logger.info(f"Error in section summarization: {str(e)}")
            return None

    async def summarize_course_from_blob(self, full_blob_path: str, subject_name: str = None,
                                         subject_type: str = None) -> str | None:
        """
        Summarize complete course from all section summary files in blob storage
        Args:
            full_blob_path: Path to section_summaries folder (e.g., "CS101/section_summaries")
            subject_name: Subject name for context
            subject_type: Subject type for context
        Returns:
            Summary path in blob or None if failed
        """

        try:
            # Parse path: "CS101/section_summaries" -> ["CS101", "section_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 2:
                logger.info(f" Invalid path: {full_blob_path}. Should be in format: CourseID/section_summaries")
                return None

            course_id = path_parts[0]  # CS101
            # path_parts[1] should be section_summaries

            logger.info(f" CourseID: {course_id}")
            logger.info(f" section_summaries path: {full_blob_path}")

            # Use the shared blob manager instance
            all_files = await self.blob_manager.list_files()

            # Filter files in section_summaries folder
            sections_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not sections_files:
                logger.info(f" No section summary files found in {full_blob_path}")
                return None

            logger.info(f" Found {len(sections_files)} section summary files:")
            for file in sections_files:
                logger.info(f"  - {file}")

            # Download and read all files directly to memory
            all_content = ""
            successful_files = []

            for file_path in sections_files:
                logger.info(f"\n Downloading file to memory: {file_path}")

                try:
                    # Download file directly to memory
                    file_bytes = await self.blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # Convert to text
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"Section: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f" File read successfully: {len(file_content)} characters")
                        else:
                            logger.info(f" Empty file: {file_path}")
                    else:
                        logger.info(f" Failed to download file: {file_path}")

                except Exception as e:
                    logger.info(f" Error processing file {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f" Could not read any files from {full_blob_path}")
                return None

            logger.info(f"\n Total working with {len(successful_files)} files")
            logger.info(f" Total content length: {len(all_content)} characters")

            # Create summary
            logger.info(f"\n Creating complete course summary...")

            # Prepare special prompt for course summary
            system_prompt = self._get_course_summary_prompt(subject_name, subject_type)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": all_content
                }
            ]

            # Call language model
            logger.info(f"Final prompt: {messages}")
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            course_summary = response.choices[0].message.content

            logger.info(f"Course summary created successfully!")
            logger.info(f" Summary length: {len(course_summary)} characters")

            # Add title to course summary
            if subject_name:
                course_title = f"# סיכום קורס {subject_name}\n\n"
            else:
                course_title = f"# סיכום קורס\n\n"
            course_summary_with_title = course_title + course_summary

            # Save summary to blob in new structure: CourseID/course_summary.md
            summary_blob_path = f"{course_id}/course_summary.md"

            logger.info(f" Saving course summary to blob: {summary_blob_path}")

            success = await self.blob_manager.upload_text_to_blob(
                text_content=course_summary_with_title,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f" Course summary saved to blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f" Failed to save course summary to blob")
                return None

        except Exception as e:
            logger.info(f" Error in course summarization: {str(e)}")
            return None


async def main():
    """Main function for testing all three types of summaries"""
    logger.info("Content Summarizer - Testing All Summary Types with Subject Parameters")
    logger.info("=" * 70)

    summarizer = ContentSummarizer()

    # Test parameters
    subject_name = "מתמטיקה בדידה"
    subject_type = "מתמטי"
    course_id = "Discrete_mathematics"
    section_id = "Section2"

    # # ========================================
    # # TEST 1: Individual File Summaries
    # # ========================================
    # logger.info("\n" + "=" * 70)
    # logger.info("TEST 1: Testing Individual File Summaries (summarize_md_file)")
    # logger.info("=" * 70)
    #
    # test_files = [
    #     f"{course_id}/{section_id}/Videos_md/2.md"
    #     # f"{course_id}/{section_id}/Docs_md/2002.md"
    # ]
    #
    # successful_files = 0
    # for i, blob_path in enumerate(test_files, 1):
    #     logger.info(f"\n--- File Test {i}/{len(test_files)} ---")
    #     logger.info(f"Testing file: {blob_path}")
    #     logger.info(f"Subject: {subject_name} ({subject_type})")
    #
    #     try:
    #         result = await summarizer.summarize_md_file(
    #             blob_path=blob_path,
    #             subject_name=subject_name,
    #             subject_type=subject_type
    #         )
    #
    #         if result:
    #             logger.info(f"File summary created successfully!")
    #             logger.info(f"Summary saved to: {result}")
    #             successful_files += 1
    #         else:
    #             logger.info(f"Failed to create file summary")
    #
    #     except Exception as e:
    #         logger.info(f" Error during file summarization: {str(e)}")
    #         traceback.print_exc()
    #
    #     if i < len(test_files):
    #         logger.info(" Waiting 3 seconds before next file...")
    #         await asyncio.sleep(3)
    #
    # logger.info(f"\nFile Summary Results: {successful_files}/{len(test_files)} successful")
    #
    # # ========================================
    # # TEST 2: Section Summary
    # # ========================================
    # logger.info("\n" + "=" * 70)
    # logger.info("TEST 2: Testing Section Summary (summarize_section_from_blob)")
    # logger.info("=" * 70)
    #
    # section_path = f"{course_id}/{section_id}/file_summaries"
    # logger.info(f"Testing section path: {section_path}")
    # logger.info(f"Subject: {subject_name} ({subject_type})")
    #
    # try:
    #     logger.info("Waiting 30 seconds before section summary...")
    #     # await asyncio.sleep(30)
    #
    #     # נתיב לסיכום הקודם
    #     previous_section_path = f"{course_id}/section_summaries/Section1.md"
    #     logger.info(f"Using previous section summary path: {previous_section_path}")
    #
    #     section_result = await summarizer.summarize_section_from_blob(
    #         full_blob_path=section_path,
    #         subject_name=subject_name,
    #         subject_type=subject_type,
    #         previous_summary_path=previous_section_path  # מעביר נתיב במקום טקסט
    #     )
    #
    #     if section_result:
    #         logger.info(f"Section summary created successfully!")
    #         logger.info(f"Section summary saved to: {section_result}")
    #     else:
    #         logger.info(f"Failed to create section summary")
    #         logger.info(f"Make sure there are file summaries in: {section_path}")
    #
    # except Exception as e:
    #     logger.info(f"Error during section summarization: {str(e)}")
    #     traceback.print_exc()

    # ========================================
    # TEST 3: Course Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Testing Course Summary (summarize_course_from_blob)")
    logger.info("=" * 70)

    course_path = f"{course_id}/section_summaries"
    logger.info(f"Testing course path: {course_path}")
    logger.info(f"Subject: {subject_name} ({subject_type})")

    try:
        # logger.info(" Waiting 5 seconds before course summary...")
        # await asyncio.sleep(30)

        course_result = await summarizer.summarize_course_from_blob(
            full_blob_path=course_path,
            subject_name=subject_name,
            subject_type=subject_type
        )

        if course_result:
            logger.info(f"Course summary created successfully!")
            logger.info(f"Course summary saved to: {course_result}")
        else:
            logger.info(f"Failed to create course summary")
            logger.info(f"Make sure there are section summaries in: {course_path}")

    except Exception as e:
        logger.info(f" Error during course summarization: {str(e)}")
        traceback.print_exc()

    # ========================================
    # FINAL SUMMARY
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTING COMPLETED - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Course: {subject_name} ({subject_type})")
    logger.info(f"Course ID: {course_id}")
    logger.info(f"Section ID: {section_id}")
    logger.info("")
    logger.info("Tests Performed:")
    # logger.info(f"   1. Individual File Summaries: {successful_files}/{len(test_files)} successful")
    # logger.info(f"   2. Section Summary: {'Correct' if 'section_result' in locals() and section_result else 'Error'}")
    logger.info(f"   3. Course Summary: {'Correct' if 'course_result' in locals() and course_result else 'Error'}")
    logger.info("")
    logger.info("Note: Section and Course summaries depend on previous summaries existing in blob storage")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
