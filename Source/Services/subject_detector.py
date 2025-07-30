"""
Subject Detection Service
Identifies subject type (mathematical/humanities) based on file and video lists
"""

import re
from typing import List, Optional, Dict
from openai import AzureOpenAI
from Source.Services.blob_manager import BlobManager
from Config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL
)
from Config.logging_config import setup_logging
logger = setup_logging()

class SubjectDetector:
    """Identifies subject type based on file and video analysis"""

    def __init__(self, max_vid: int = 5, max_doc: int = 5):
        self.blob_manager = BlobManager(container_name="processeddata")
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.max_vid = max_vid
        self.max_doc = max_doc
        logger.info(f"🔧 File limits: maximum {self.max_vid} videos, {self.max_doc} documents")

    def extract_subject_type_from_video_md(self, md_content: str) -> Optional[str]:
        """
        Extract subject type from video markdown file

        Args:
            md_content: Markdown file content

        Returns:
            "מתמטי" or "הומני" or None if not found
        """
        logger.info("🎬 Starting subject type extraction from video markdown file")
        # Search for the "Subject Type" section
        pattern = r'## 🎓 סוג מקצוע\s*\n\s*([^\n]+)'
        match = re.search(pattern, md_content)

        if match:
            subject_type = match.group(1).strip()
            logger.info(f"✅ Found subject type: {subject_type}")
            if subject_type in ['מתמטי', 'הומני']:
                return subject_type

        logger.info("⚠️ Subject type not found in video file")
        return None

    def extract_full_transcript_from_video_md(self, md_content: str) -> str:
        """
        Extract full transcript from video markdown file

        Args:
            md_content: Markdown file content

        Returns:
            Full transcript or empty string
        """
        logger.info("📄 Starting full transcript extraction from video file")
        # Search for "Full Transcript" section
        pattern = r'## 📄 טרנסקריפט מלא\s*\n(.*?)(?=\n## |\n$)'
        match = re.search(pattern, md_content, re.DOTALL)

        if match:
            transcript = match.group(1).strip()
            logger.info(f"✅ Found transcript with length {len(transcript)} characters")
            return transcript

        logger.info("⚠️ No transcript found in video file")
        return ""

    def analyze_files_with_llm(self, file_contents: List[Dict[str, str]]) -> str:
        """
        Analyze files using language model to determine subject type
        Limits number of files according to max_vid and max_doc

        Args:
            file_contents: List of dictionaries with 'path' and 'content' of each file

        Returns:
            "מתמטי" or "הומני"
        """
        logger.info(f"🤖 Starting analysis with language model for {len(file_contents)} files")
        logger.info(f"📏 Limits: maximum {self.max_vid} videos, {self.max_doc} documents")

        # Separate files into videos and documents
        video_files = []
        doc_files = []

        for file_info in file_contents:
            if '/Videos_md/' in file_info['path']:
                video_files.append(file_info)
            else:
                doc_files.append(file_info)

        logger.info(f"  📊 Found: {len(video_files)} videos, {len(doc_files)} documents")

        # Limit number of files
        selected_videos = video_files[:self.max_vid]
        selected_docs = doc_files[:self.max_doc]

        if len(video_files) > self.max_vid:
            logger.info(f"  ⚠️ Limited videos to {self.max_vid} out of {len(video_files)}")
        if len(doc_files) > self.max_doc:
            logger.info(f"  ⚠️ Limited documents to {self.max_doc} out of {len(doc_files)}")

        # Combine selected files
        selected_files = selected_videos + selected_docs

        logger.info(f"  ✅ Analyzing {len(selected_files)} files: {len(selected_videos)} videos + {len(selected_docs)} documents")

        # יצירת פרומפט עם הקבצים הנבחרים
        prompt = """אתה מומחה בסיווג תוכן אקדמי. עליך לנתח את התוכן הבא ולקבוע האם זה מקצוע מתמטי/טכני או הומני.

קריטריונים לסיווג:
- מתמטי: מתמטיקה, פיזיקה, מדעי המחשב, הנדסה, סטטיסטיקה, לוגיקה, אלגוריתמים
- הומני: ספרות, היסטוריה, פילוסופיה, פסיכולוגיה, סוציולוגיה, אמנות, שפות

תוכן הקבצים לניתוח:

"""

        for i, file_info in enumerate(selected_files, 1):
            logger.info(f"  📄 Adding file {i} to analysis: {file_info['path']}")
            prompt += f"\n--- קובץ {i}: {file_info['path']} ---\n"

            # If it's a video, use only the full transcript
            if '/Videos_md/' in file_info['path']:
                transcript = self.extract_full_transcript_from_video_md(file_info['content'])
                content = transcript if transcript else file_info['content']
                logger.info(f"    🎬 Video - transcript length: {len(content)} characters")
            else:
                # For documents, take all content
                content = file_info['content']
                logger.info(f"    📋 Document - content length: {len(content)} characters")

            prompt += content
            prompt += "\n" + "="*50 + "\n"

        prompt += """
על בסיס התוכן שניתח, השב במילה אחת בלבד:
- "מתמטי" אם זה מקצוע מתמטי/טכני
- "הומני" אם זה מקצוע הומני

תשובה:"""

        logger.info(f"  📊 Total prompt length: {len(prompt)} characters")
        logger.info(f"  🔄 Sending request to language model...")

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_CHAT_COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": "אתה מומחה בסיווג תוכן אקדמי. תמיד השב במילה אחת בלבד: 'מתמטי' או 'הומני'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"  🎯 Model response: '{result}'")

            # Validate response is correct
            if result in ['מתמטי', 'הומני']:
                logger.info(f"  ✅ Valid response: {result}")
                return result
            else:
                logger.info(f"  ⚠️ Unexpected response from model: {result}, returning 'לא זוהה'")
                return "לא זוהה"

        except Exception as e:
            logger.info(f"  ❌ Error in analysis with language model: {e}")
            return "לא זוהה"

    def detect_subject_from_course_path(self, course_path: str) -> str:
        """
        Identify subject type based on course blob path

        Logic:
        1. Find all video and document files in course
        2. If there are more than 2 video files with same subject_type - return it
        3. Otherwise - pass to language model for analysis of all files

        Args:
            course_path: Course path in blob storage (e.g., "CS101")

        Returns:
            "מתמטי", "הומני", or "לא זוהה"
        """
        logger.info(f"🎓 Starting subject type identification for course: {course_path}")

        # Find all files in course
        all_files = self.blob_manager.list_files()

        # Filter files that start with course_path and end with .md
        course_files = [
            f for f in all_files
            if f.startswith(course_path + "/") and f.endswith(".md") and
            ("/Videos_md/" in f or "/Docs_md/" in f)
        ]

        if not course_files:
            logger.info(f"  ❌ No files found in course: {course_path}")
            return "לא זוהה"

        logger.info(f"  📁 Found {len(course_files)} files in course:")
        for file in course_files:
            logger.info(f"    - {file}")

        # Call existing function with file list
        return self.detect_subject(course_files)

    def detect_subject(self, file_paths: List[str]) -> str:
        """
        Identify subject type based on file list

        Logic:
        1. If all videos have same subject_type - return it
        2. Otherwise - pass to language model for analysis

        Args:
            file_paths: List of file paths in blob storage

        Returns:
            "מתמטי", "הומני", or "לא זוהה"
        """
        logger.info(f"🔍 Identifying subject type for {len(file_paths)} files")

        video_subject_types = []
        file_contents = []

        # Process each file
        for file_path in file_paths:
            logger.info(f"  📄 Processing file: {file_path}")

            # Download file content
            content = self.blob_manager.download_to_memory(file_path)
            if not content:
                logger.info(f"    ⚠️ Cannot download file: {file_path}")
                continue

            try:
                md_content = content.decode('utf-8')
                file_contents.append({
                    'path': file_path,
                    'content': md_content
                })

                # If it's a video file, try to extract subject_type
                if '/Videos_md/' in file_path:
                    subject_type = self.extract_subject_type_from_video_md(md_content)
                    if subject_type:
                        video_subject_types.append(subject_type)
                        logger.info(f"    ✅ Found subject type: {subject_type}")
                    else:
                        logger.info(f"    ⚠️ No subject type found in video")

            except UnicodeDecodeError:
                logger.info(f"    ❌ Error reading file: {file_path}")
                continue

        # Check if all videos have same subject_type (only if there are at least 2 videos)
        if len(video_subject_types) >= 2:
            unique_types = list(set(video_subject_types))
            logger.info(f"  📊 Found subject types in videos: {video_subject_types}")

            if len(unique_types) == 1:
                result = unique_types[0]
                logger.info(f"  ✅ All videos ({len(video_subject_types)}) have same subject type: {result}")
                return result
            else:
                logger.info(f"  🔄 Found different subject types in videos, passing to language model")
        elif len(video_subject_types) == 1:
            logger.info(f"  ⚠️ Found only one video with subject type: {video_subject_types[0]}, passing to language model for validation")
        else:
            logger.info(f"  🔄 No videos with defined subject type found, passing to language model")

        # If no agreement or no videos with subject_type, use language model
        if not file_contents:
            logger.info("  ❌ No content for analysis")
            return "לא זוהה"

        logger.info(f"  🤖 Analyzing {len(file_contents)} files with language model...")
        result = self.analyze_files_with_llm(file_contents)
        logger.info(f"  ✅ Language model analysis result: {result}")

        return result


def detect_subject_from_paths(file_paths: List[str], max_vid: int = 5, max_doc: int = 5) -> str:
    """
    Convenient function for subject type identification from file list

    Args:
        file_paths: List of file paths in blob storage
        max_vid: Maximum number of video files for analysis
        max_doc: Maximum number of document files for analysis

    Returns:
        "מתמטי", "הומני", or "לא זוהה"
    """
    logger.info(f"🚀 Starting subject type identification for {len(file_paths)} files")
    detector = SubjectDetector(max_vid=max_vid, max_doc=max_doc)
    return detector.detect_subject(file_paths)


def detect_subject_from_course(course_path: str, max_vid: int = 5, max_doc: int = 5) -> str:
    """
    Convenient function for subject type identification from course path

    Args:
        course_path: Course path in blob storage (e.g., "CS101")
        max_vid: Maximum number of video files for analysis
        max_doc: Maximum number of document files for analysis

    Returns:
        "מתמטי", "הומני", or "לא זוהה"
    """
    logger.info(f"🎯 Starting subject type identification for course: {course_path}")
    detector = SubjectDetector(max_vid=max_vid, max_doc=max_doc)
    return detector.detect_subject_from_course_path(course_path)


if __name__ == "__main__":
    # Function testing
    logger.info("🧪 Subject type identification testing")
    logger.info("=" * 50)

    # Test 1: Identification from full course (new function)
    logger.info("\n🔍 Test 1: Subject type identification from full course")
    logger.info("-" * 40)

    course_path = "CS101"

    try:
        result = detect_subject_from_course(course_path)
        logger.info(f"🎯 Result for course {course_path}: {result}")
    except Exception as e:
        logger.info(f"❌ Error in course testing: {e}")
        import traceback
        traceback.logger.info_exc()

    logger.info("\n✅ Tests completed!")
