"""
Content Summarizer - Summarization system for videos and documents
Uses Azure OpenAI language model to create customized summaries
"""

import os
import asyncio
import traceback
from typing import Dict
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

    def __init__(self):
        """
        Initialize summarization system

        Args:
            model_name: Model name in Azure OpenAI (default: gpt-4o)
        """
        self.model_name = AZURE_OPENAI_CHAT_COMPLETION_MODEL

        # Create async OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # Create BlobManager for accessing files in blob storage
        self.blob_manager = BlobManager()
        logger.info(f"ContentSummarizer initialized with model: {self.model_name}")

    def _get_video_summary_prompt(self, subject_type: str = None, existing_summary: str = None) -> str:
        """Prepare prompt for video summarization with adaptation to subject type and existing summary if available"""

        # Clear opening — identity and role
        base_prompt = (
            "אתה מומחה לסיכום שיעורים אקדמיים. "
            "קיבלת תמליל מלא של הרצאת וידאו באורך כשעתיים."
        )

        # Addition if existing summary exists
        if existing_summary:
            base_summary = f"""

    סיכום קיים:
    {existing_summary}

    שים לב: הסיכום הקיים הוא רק נקודת פתיחה — המטרה שלך היא להרחיב ולפרט אותו משמעותית בהתבסס על כל הטרנסקריפט.
    אל תחסוך בפרטים — הוסף דוגמאות, הסברים והערות נוספות כדי להפוך אותו לסיכום מקיף ומלא.
    """
        else:
            base_summary = ""

        # Special instructions by subject type if available
        if subject_type == "מתמטי":
            specific_instructions = """

    זהו קורס מתמטי:
    - הדגש נוסחאות, הגדרות מתמטיות ומשפטים.
    - כלול דוגמאות מספריות ופתרונות שלב-שלב.
    - הסבר את הלוגיקה מאחורי הוכחות.
    - פרט כל נוסחה ופונקציה שמוזכרת.
    - שמור על דיוק מתמטי וסימונים נכונים.
    """
        elif subject_type == "הומני":
            specific_instructions = """

    זהו קורס הומני:
    - הדגש רעיונות מרכזיים, תיאוריות וגישות חשיבה.
    - כלול דוגמאות מהחיים, מקרי מבחן והקשרים היסטוריים.
    - ציין דעות שונות ומחלוקות אם רלוונטי.
    - עזור להבין מושגים מופשטים בצורה ברורה.
    """
        else:
            specific_instructions = ""

        # Main instructions and structure
        main_instructions = """

    המטרה שלך:
    - ליצור סיכום מפורט, מקיף ופדגוגי שמאפשר לסטודנט להבין את כל החומר גם בלי לצפות בהרצאה.
    - סדר את הסיכום לפי הרצף הכרונולוגי של השיעור.
    - אל תחסוך בפרטים — השתמש בכמה טוקנים שצריך כדי שהסיכום יהיה שלם ואינפורמטיבי, גם אם הוא יוצא ארוך מאוד.

    מבנה הפלט:
    1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.
    2. **סיכום מפורט של השיעור** — כולל הסברים, דוגמאות והערות של המרצה, כתוב בשפה ברורה ונגישה.
    3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.

    זכור:
    - כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.
    - סדר את הכל בצורה שתשקף את הזרימה המקורית של ההרצאה.

    התמליל:
    """

        # Combine everything
        return base_prompt + base_summary + specific_instructions + main_instructions

    def _get_document_summary_prompt(self) -> str:
        """Prepare prompt for document summarization"""
        return """אתה מומחה לסיכום חומרי לימוד אקדמיים. קיבלת מסמך לימוד מתוך קורס אוניברסיטאי.
המסמך יכול להיות כל סוג של חומר: סיכום נושא, דף נוסחאות, דף תרגול, פתרון או כל חומר לימודי אחר.
עליך לזהות את סוג התוכן ולהתאים את הסיכום בצורה שתשרת את הסטודנט בצורה הטובה ביותר.

המטרה שלך:
- ליצור סיכום מקיף, מפורט וברור, שמכסה *את כל* החומר, כולל דוגמאות והערות של המרצה.
- כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.
- סדר את הסיכום לפי הרצף הכרונולוגי של השיעור.

המשימה שלך:
- זהה את סוג התוכן והתאם את אופי הסיכום והמבנה בהתאם.
- שמור על הסדר הלוגי שבו מוצג החומר במסמך.
- פרט והסבר מושגים, דוגמאות, כללים או נוסחאות, לפי הצורך.
- הדגש נקודות חשובות והקשרים בין רעיונות או נושאים.

מבנה הפלט (מותאם לסוג המסמך):
- התחלה קצרה: כתוב משפט או שתיים שמסבירים מה מכיל הסיכום.
- חלק מרכזי: סיכום מפורט של החומר עם כל ההסברים הנדרשים.
-  במידת הצורך:: רשימת נקודות עיקריות או המלצות לחזרה ותרגול, אם רלוונטי.

זכור:
- סדר את התוכן באופן ברור והגיוני.
- שמור על טון מסביר ופשוט להבנה.
- הקפד להתאים את מבנה הסיכום לסוג הקובץ שקיבלת.

המסמך:
"""

    def _get_section_summary_prompt(self) -> str:
        """Prepare prompt for complete Section summarization"""
        return """אתה מומחה לסיכום חומרי לימוד אקדמיים. קיבלת אוסף של סיכומים כתובים (Markdown) מתוך Section שלם בקורס אוניברסיטאי.
    כל סיכום מייצג שיעור, מסמך או תרגול שנלמדו באותו Section.  
    המטרה שלך היא לאחד את כל הסיכומים לכדי סיכום-על **מפורט**, מקיף ופדגוגי שמציג את התמונה הכוללת של ה-Section.

    זכור: המטרה שלך **אינה לקצר** את החומר אלא לארגן אותו מחדש, להרחיב ולהסביר כך שהסטודנט יוכל ללמוד את כל החומר מתוך הסיכום הסופי **ללא תלות בחומרים המקוריים**. אל תחסוך בפרטים — ציין הגדרות, דוגמאות, הסברים והערות חשובות שהיו בפירוטי הקבצים שניתנו.

    המטרה שלך:
    - ליצור סיכום מקיף של כל ה-Section שמכסה את כל החומרים שקיבלת.
    - לזהות קשרים ונושאים משותפים בין הקבצים השונים.
    - לסדר את החומר בצורה לוגית ומובנת.
    - ליצור מבט כולל על כל הנושאים שנלמדו ב-Section.

    המשימה שלך:
    - פתח את הסיכום במשפט או שניים שמציגים בקצרה מה נלמד בסקשן ומה המטרה שלו.
    - עבור על כל הקבצים וזהה את הנושאים העיקריים.
    - מצא קשרים והמשכיות בין הנושאים השונים.
    - סדר את החומר בצורה הגיונית - מהבסיסי למתקדם או לפי רצף הלמידה.
    - הדגש נקודות חשובות, מושגי מפתח ודגשים שחוזרים על עצמם.

    מבנה הפלט:
    1. **פתיח קצר** — משפט או שניים שמסבירים מה נלמד ומה מטרת הסקשן.
    2. **סקירה כללית של ה-Section** — רשימה מסודרת של הנושאים המרכזיים.
    3. **סיכום מפורט לפי נושאים** — חלוקה לוגית של החומר עם הסברים מקיפים, דוגמאות והבהרות.
    4. **נקודות מפתח והמלצות ללמידה** — דגשים חשובים לזכירה ודרכי פעולה לחזרה ותרגול.

    זכור:
    - שמור על מבנה מסודר והגיוני שמקל על הבנה.
    - כתוב בצורה ברורה, נגישה ומלווה — כאילו אתה מדריך את הסטודנט שלב אחר שלב.
    - אל תדלג על פרטים חשובים — המטרה היא סיכום שלם ומקיף.

    סיכומי כל הקבצים:
    """

    def _get_course_summary_prompt(self) -> str:
        """Prepare prompt for reorganizing complete course content"""
        return """אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים. קיבלת אוסף של סיכומי Section מתוך קורס אוניברסיטאי שלם.
        כל סיכום Section מייצג חלק משמעותי מהחומר, שכבר עבר עיבוד מפורט. כעת תפקידך הוא לשלב, לארגן ולהציג מחדש את התוכן הקיים בצורה **מלאה**, **ברורה** ו**פדגוגית** — כך שסטודנט יוכל ללמוד את כל חומר הקורס מתוך תוצר אחד כולל.

        שים לב: המשימה **אינה לקצר** את החומר או להשמיט פרטים, אלא לבנות מבנה כולל, ברור ומקושר של כל תוכן הקורס. 
        עליך **לשלב באופן פעיל דוגמאות, הסברים, הגדרות והערות** – אלה לא רק חלק מהחומר, אלא כלים מרכזיים להבנתו. 
        הדוגמאות בפרט הן חלק בלתי נפרד מהלמידה – השתמש בהן כדי להמחיש מושגים, להעמיק את ההבנה, ולחבר את הסטודנט לחומר ברמה יישומית.

        המטרה שלך:
        - ליצור הצגה חינוכית של הקורס כולו, המבוססת על כלל ה-Sections שסופקו.
        - לזהות את המבנה הלוגי וההתפתחות הפדגוגית של הקורס.
        - לארגן את החומר באופן שמדגיש את ההתקדמות מהבסיס למתקדם.
        - להדגיש חיבורים וקשרים בין נושאים המופיעים לאורך הקורס.

        המשימה שלך:
        - עבור על כל סיכומי ה-Sections והבין מהם הרעיונות המרכזיים של הקורס.
        - גלה את ההתפתחות המושגית והלוגית לאורך ה-Sections.
        - סדר את התוכן כך שישקף את זרימת ההוראה כפי שהייתה בקורס עצמו.
        - **שמור על כל עומק, הסבר ודוגמה רלוונטית** – אל תדלג על שום פרט שיכול לתרום ללמידה.
        - הדגש מושגים חוזרים, מעמיקים ומתפתחים לאורך הקורס.

        הצגת הקורס:
        """

    async def parse_video_md_file_from_blob(self, blob_path: str) -> Dict:
        """
        Parse video.md file from blob storage into its specific parts

        Args:
            blob_path: Path to video.md file in blob storage

        Returns:
            Dictionary with different parts of the file
        """
        logger.info(f"Parsing video MD file from blob: {blob_path}")

        # Download file from blob storage
        file_bytes = await self.blob_manager.download_to_memory(blob_path)
        if not file_bytes:
            raise FileNotFoundError(f"File not found in blob: {blob_path}")

        content = file_bytes.decode('utf-8')

        # Search for specific parts
        subject_type = None
        existing_summary = None
        full_transcript = None

        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line_stripped = line.strip()

            # Identify section beginnings
            if line_stripped == "## סוג מקצוע":
                if current_section and section_content:
                    # Save previous section
                    if current_section == "subject_type":
                        subject_type = '\n'.join(section_content).strip()
                    elif current_section == "existing_summary":
                        existing_summary = '\n'.join(section_content).strip()
                    elif current_section == "full_transcript":
                        full_transcript = '\n'.join(section_content).strip()

                current_section = "subject_type"
                section_content = []

            elif line_stripped == "## סיכום השיעור":
                if current_section and section_content:
                    if current_section == "subject_type":
                        subject_type = '\n'.join(section_content).strip()

                current_section = "existing_summary"
                section_content = []

            elif line_stripped in ["## Full Transcript", "## טרנסקריפט מלא", "## טרנסקריפט מלא"]:
                if current_section and section_content:
                    if current_section == "existing_summary":
                        existing_summary = '\n'.join(section_content).strip()

                current_section = "full_transcript"
                section_content = []

            elif line_stripped.startswith("## ") and current_section == "full_transcript":
                # End transcript when reaching new section
                if section_content:
                    full_transcript = '\n'.join(section_content).strip()
                break

            else:
                # Add content to current section
                if current_section:
                    section_content.append(line)

        # Save last section
        if current_section and section_content:
            if current_section == "subject_type":
                subject_type = '\n'.join(section_content).strip()
            elif current_section == "existing_summary":
                existing_summary = '\n'.join(section_content).strip()
            elif current_section == "full_transcript":
                full_transcript = '\n'.join(section_content).strip()


        # logger.info(f" Detected subject type: {subject_type}")
        # logger.info(f" Existing summary length: {len(existing_summary) if existing_summary else 0} chars")
        # logger.info(f" Full transcript length: {len(full_transcript) if full_transcript else 0} chars")
        #
        # # הדפסת תוכן מפורט
        # logger.info("\n" + "=" * 60)
        # logger.info(" PARSED CONTENT DETAILS:")
        # logger.info("=" * 60)
        #
        # logger.info(f"\n SUBJECT TYPE:")
        # if subject_type:
        #     logger.info(f"'{subject_type}'")
        # else:
        #     logger.info("None")
        #
        # logger.info(f"\n EXISTING SUMMARY:")
        # if existing_summary:
        #     logger.info(f"'{existing_summary}'")
        # else:
        #     logger.info("None")

        # logger.info(f"\n TRANSCRIPT PREVIEW:")
        # if full_transcript:
        #     lines = full_transcript.split('\n')
        #     if len(lines) > 4:
        #         logger.info("First 2 lines:")
        #         for i, line in enumerate(lines[:2]):
        #             logger.info(f"  {i + 1}: {line}")
        #         logger.info("  ...")
        #         logger.info("Last 2 lines:")
        #         for i, line in enumerate(lines[-2:], len(lines) - 1):
        #             logger.info(f"  {i}: {line}")
        #     else:
        #         logger.info("Full transcript (short):")
        #         for i, line in enumerate(lines):
        #             logger.info(f"  {i + 1}: {line}")
        # else:
        #     logger.info("None")
        #
        # logger.info("=" * 60)

        return {
            "subject_type": subject_type,
            "existing_summary": existing_summary,
            "full_transcript": full_transcript,
            "original_content": content
        }


    async def summarize_content(self, content: str, content_type: str = "document", subject_type: str = None, existing_summary: str = None) -> str:
        """
        Create summary for content

        Args:
            content: Content to summarize (MD text)
            content_type: Content type - "video" or "document"
            subject_type: Subject type (for video only)
            existing_summary: Existing summary (for video only)

        Returns:
            Generated summary
        """
        logger.info(f"Creating summary for {content_type} content...")
        logger.info(f"Content length: {len(content)} characters")

        try:
            # Choose prompt by content type
            if content_type.lower() == "video":
                logger.info(f"Subject type: {subject_type}")
                logger.info(f"Has existing summary: {bool(existing_summary)}")
                system_prompt = self._get_video_summary_prompt(
                    subject_type=subject_type,
                    existing_summary=existing_summary
                )
            else:
                system_prompt = self._get_document_summary_prompt()

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
        if "Videos_md" in blob_path.lower():
            return "video"
        elif "Docs_md" in blob_path.lower():
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

    async def summarize_md_file(self, blob_path: str) -> str | None:
        """
        Summarize MD file from blob with automatic content type detection and save to blob

        Args:
            blob_path: Path to MD file in blob

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
                        subject_type=parsed_data.get("subject_type"),
                        existing_summary=parsed_data.get("existing_summary")
                    )

                # If it's a regular document - standard handling
                else:
                    logger.info("Document file - using standard processing")

                    if not content.strip():
                        logger.info(f"File is empty")
                        return None

                    # Create summary
                    summary = await self.summarize_content(content, content_type)

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

            # Create new summary path
            summary_blob_path = f"{course_id}/{section_id}/file_summaries/{base_name}.md"

            logger.info(f"Saving summary to blob: {summary_blob_path}")

            # Save to blob
            success = await self.blob_manager.upload_text_to_blob(
                text_content=summary,
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

    async def summarize_section_from_blob(self, full_blob_path: str) -> str | None:
        """
        Summarize complete section from all summary files in blob storage
        Args:
            full_blob_path: Path to file_summaries folder (e.g., "CS101/Section1/file_summaries")
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

            # Create BlobManager with default container
            blob_manager = BlobManager()

            # Get list of all files in container
            all_files = await blob_manager.list_files()

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
                    file_bytes = await blob_manager.download_to_memory(file_path)

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

            # Create summary
            logger.info(f"\n Creating section summary...")

            # Prepare special prompt for section summary
            system_prompt = self._get_section_summary_prompt()

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
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            section_summary = response.choices[0].message.content

            logger.info(f" Section summary created successfully!")
            logger.info(f" Summary length: {len(section_summary)} characters")

            # Save summary to blob in new structure: CourseID/section_summaries/SectionID.md
            summary_blob_path = f"{course_id}/section_summaries/{section_id}.md"

            logger.info(f"Saving section summary to blob: {summary_blob_path}")

            success = await blob_manager.upload_text_to_blob(
                text_content=section_summary,
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


    async def summarize_course_from_blob(self, full_blob_path: str) -> str | None:
        """
        Summarize complete course from all section summary files in blob storage
        Args:
            full_blob_path: Path to section_summaries folder (e.g., "CS101/section_summaries")
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

            # Create BlobManager with default container
            blob_manager = BlobManager()

            # Get list of all files in container
            all_files = await blob_manager.list_files()

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
                    file_bytes = await blob_manager.download_to_memory(file_path)

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
            system_prompt = self._get_course_summary_prompt()

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
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            course_summary = response.choices[0].message.content

            logger.info(f"Course summary created successfully!")
            logger.info(f" Summary length: {len(course_summary)} characters")

            # Save summary to blob in new structure: CourseID/course_summary.md
            summary_blob_path = f"{course_id}/course_summary.md"

            logger.info(f" Saving course summary to blob: {summary_blob_path}")

            success = await blob_manager.upload_text_to_blob(
                text_content=course_summary,
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
    """Main function for testing"""
    logger.info("Content Summarizer - Testing")
    logger.info("=" * 50)

    summarizer = ContentSummarizer()

    # logger.info("\n Testing summarize_md_file with blob paths...")
    #
    # test_blob_paths = [
    #     "CS101/Section1/Videos_md/2.md",
    #     "CS101/Section1/Docs_md/1.md",
    # ]
    #
    # successful_tests = 0
    # failed_tests = 0
    #
    # for blob_path in test_blob_paths:
    #     logger.info(f"\n{'=' * 50}")
    #     logger.info(f" Testing blob: {blob_path}")
    #     logger.info(f"{'=' * 50}")
    #
    #     try:
    #         # בדיקה אם הקובץ קיים בבלוב
    #         logger.info(f" Checking if blob exists...")
    #
    #         # יצירת סיכום מהבלוב
    #         summary = summarizer.summarize_md_file(blob_path)
    #
    #         if summary and not summary.startswith("שגיאה") and not summary.startswith(
    #                 "לא ניתן") and not summary.startswith("נכשל"):
    #             logger.info(f" Summary created successfully!")
    #             logger.info(f"Summary length: {len(summary)} characters")
    #             logger.info(f"Summary preview (first 300 chars):")
    #             logger.info("-" * 40)
    #             logger.info(summary[:300] + "..." if len(summary) > 300 else summary)
    #             logger.info("-" * 40)
    #
    #             # שמירת הסיכום לקובץ מקומי
    #             summary_file = summarizer.save_summary_to_file(summary, blob_path, "summaries")
    #             if summary_file:
    #                 logger.info(f" Summary saved to: {summary_file}")
    #
    #             successful_tests += 1
    #
    #         else:
    #             logger.info(f"Failed to create summary: {summary}")
    #             failed_tests += 1
    #
    #     except Exception as e:
    #         logger.info(f" Error processing blob {blob_path}: {str(e)}")
    #         failed_tests += 1
    #
    #     logger.info(f"\n Waiting 2 seconds before next test...")
    #     import time
    #     time.sleep(2)


    # # בדיקת הפונקציה summarize_section_from_blob
    # logger.info("\n Testing summarize_section_from_blob...")
    #
    # # נתיב מלא בבלוב
    # full_blob_path = "CS101/Section1/file_summaries"
    #
    #
    # logger.info(f"Testing full blob path: {full_blob_path}")
    #
    # try:
    #     # יצירת סיכום section
    #     result = summarizer.summarize_section_from_blob(full_blob_path)
    #
    #     if result:
    #         logger.info(f"\nSection summary created successfully!")
    #         logger.info(f" Summary saved to blob: {result}")
    #         logger.info(f"Test completed successfully!")
    #     else:
    #         logger.info(f"\nFailed to create section summary")
    #         logger.info(f" Check if there are summary files in {full_blob_path}")
    #
    # except Exception as e:
    #     logger.info(f"\nError during section summarization: {str(e)}")
    #     traceback.logger.info_exc()

    # Test summarize_course_from_blob function
    logger.info("\n Testing summarize_course_from_blob...")

    # Full path to section summaries folder
    full_blob_path = "Discrete_mathematics/section_summaries"

    logger.info(f"Testing course summary from path: {full_blob_path}")

    try:
        # Create complete course summary
        result = await summarizer.summarize_course_from_blob(full_blob_path)

        if result:
            logger.info(f"\n Course summary created successfully!")
            logger.info(f"Summary saved to blob: {result}")
            logger.info(f"Test completed successfully!")
        else:
            logger.info(f"\n Failed to create course summary")
            logger.info(f"Check if there are section summary files in {full_blob_path}")

    except Exception as e:
        logger.info(f"\n Error during course summarization: {str(e)}")
        traceback.logger.info_exc()

    logger.info(f"\n Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
