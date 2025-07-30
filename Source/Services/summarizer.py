"""
Content Summarizer - מערכת סיכומים לוידאו ומסמכים
משתמשת במודל השפה מ-Azure OpenAI ליצירת סיכומים מותאמים
"""

import os
import traceback
from typing import Dict
from openai import AzureOpenAI
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
    מערכת סיכומים לתוכן - וידאו ומסמכים
    """

    def __init__(self):
        """
        אתחול מערכת הסיכומים

        Args:
            model_name: שם המודל ב-Azure OpenAI (ברירת מחדל: gpt-4o)
        """
        self.model_name = AZURE_OPENAI_CHAT_COMPLETION_MODEL

        # יצירת OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # יצירת BlobManager לגישה לקבצים ב-blob storage
        self.blob_manager = BlobManager()
        logger.info(f"✅ ContentSummarizer initialized with model: {self.model_name}")

    def _get_video_summary_prompt(self, subject_type: str = None, existing_summary: str = None) -> str:
        """הכנת prompt לסיכום וידאו עם התאמה לסוג המקצוע וסיכום בסיסי אם קיים"""

        # פתיחה ברורה — זהות ותפקיד
        base_prompt = (
            "אתה מומחה לסיכום שיעורים אקדמיים. "
            "קיבלת תמליל מלא של הרצאת וידאו באורך כשעתיים."
        )

        # תוספת אם קיים סיכום בסיסי
        if existing_summary:
            base_summary = f"""

    סיכום קיים:
    {existing_summary}

    שים לב: הסיכום הקיים הוא רק נקודת פתיחה — המטרה שלך היא להרחיב ולפרט אותו משמעותית בהתבסס על כל הטרנסקריפט.
    אל תחסוך בפרטים — הוסף דוגמאות, הסברים והערות נוספות כדי להפוך אותו לסיכום מקיף ומלא.
    """
        else:
            base_summary = ""

        # הנחיות מיוחדות לפי סוג מקצוע אם יש
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

        # ההנחיות העיקריות והמבנה
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

        # מחבר את הכל
        return base_prompt + base_summary + specific_instructions + main_instructions

    def _get_document_summary_prompt(self) -> str:
        """הכנת prompt לסיכום מסמך"""
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
        """הכנת prompt לסיכום Section שלם"""
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
        """הכנת prompt לארגון מחדש של תוכן קורס שלם"""
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

    def parse_video_md_file(self, md_file_path: str) -> Dict:
        """
        פירוק קובץ video.md לחלקים הספציפיים שלו

        Args:
            md_file_path: נתיב לקובץ video.md

        Returns:
            מילון עם החלקים השונים של הקובץ
        """
        logger.info(f"📖 Parsing video MD file: {md_file_path}")

        if not os.path.exists(md_file_path):
            raise FileNotFoundError(f"קובץ לא נמצא: {md_file_path}")

        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # חיפוש החלקים הספציפיים
        subject_type = None
        existing_summary = None
        full_transcript = None

        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line_stripped = line.strip()

            # זיהוי תחילת סקציות
            if line_stripped == "## 🎓 סוג מקצוע":
                if current_section and section_content:
                    # שמירת הסקציה הקודמת
                    if current_section == "subject_type":
                        subject_type = '\n'.join(section_content).strip()
                    elif current_section == "existing_summary":
                        existing_summary = '\n'.join(section_content).strip()
                    elif current_section == "full_transcript":
                        full_transcript = '\n'.join(section_content).strip()

                current_section = "subject_type"
                section_content = []

            elif line_stripped == "## 📝 סיכום השיעור":
                if current_section and section_content:
                    if current_section == "subject_type":
                        subject_type = '\n'.join(section_content).strip()

                current_section = "existing_summary"
                section_content = []

            elif line_stripped in ["## 📄 Full Transcript", "## 📄 טרנסקריפט מלא", "## טרנסקריפט מלא"]:
                if current_section and section_content:
                    if current_section == "existing_summary":
                        existing_summary = '\n'.join(section_content).strip()

                current_section = "full_transcript"
                section_content = []

            elif line_stripped.startswith("## ") and current_section == "full_transcript":
                # סיום הטרנסקריפט כשמגיעים לסקציה חדשה
                if section_content:
                    full_transcript = '\n'.join(section_content).strip()
                break

            else:
                # הוספת התוכן לסקציה הנוכחית
                if current_section:
                    section_content.append(line)

        # שמירת הסקציה האחרונה
        if current_section and section_content:
            if current_section == "subject_type":
                subject_type = '\n'.join(section_content).strip()
            elif current_section == "existing_summary":
                existing_summary = '\n'.join(section_content).strip()
            elif current_section == "full_transcript":
                full_transcript = '\n'.join(section_content).strip()

        # logger.info(f"🔍 Detected subject type: {subject_type}")
        # logger.info(f"📝 Existing summary length: {len(existing_summary) if existing_summary else 0} chars")
        # logger.info(f"📄 Full transcript length: {len(full_transcript) if full_transcript else 0} chars")
        #
        # # הדפסת תוכן מפורט
        # logger.info("\n" + "=" * 60)
        # logger.info("📋 PARSED CONTENT DETAILS:")
        # logger.info("=" * 60)
        #
        # logger.info(f"\n🎓 SUBJECT TYPE:")
        # if subject_type:
        #     logger.info(f"'{subject_type}'")
        # else:
        #     logger.info("None")
        #
        # logger.info(f"\n📝 EXISTING SUMMARY:")
        # if existing_summary:
        #     logger.info(f"'{existing_summary}'")
        # else:
        #     logger.info("None")

        # logger.info(f"\n📄 TRANSCRIPT PREVIEW:")
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

    def summarize_content(self, content: str, content_type: str = "document", subject_type: str = None, existing_summary: str = None) -> str:
        """
        יצירת סיכום לתוכן

        Args:
            content: התוכן לסיכום (טקסט MD)
            content_type: סוג התוכן - "video" או "document"
            subject_type: סוג המקצוע (רק לוידאו)
            existing_summary: סיכום קיים (רק לוידאו)

        Returns:
            הסיכום שנוצר
        """
        logger.info(f"📝 Creating summary for {content_type} content...")
        logger.info(f"📊 Content length: {len(content)} characters")

        try:
            # בחירת prompt לפי סוג התוכן
            if content_type.lower() == "video":
                logger.info(f"🎓 Subject type: {subject_type}")
                logger.info(f"📝 Has existing summary: {bool(existing_summary)}")
                system_prompt = self._get_video_summary_prompt(
                    subject_type=subject_type,
                    existing_summary=existing_summary
                )
            else:
                system_prompt = self._get_document_summary_prompt()

            # הכנת ההודעות
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

            # קריאה למודל השפה
            logger.info(f"🤖 Calling {self.model_name} for summarization...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # יציבות בסיכום
                top_p=0.7
            )

            summary = response.choices[0].message.content

            logger.info(f"✅ Summary created successfully!")
            logger.info(f"📊 Summary length: {len(summary)} characters")

            return summary

        except Exception as e:
            logger.info(f"❌ Error creating summary: {e}")
            return f"שגיאה ביצירת סיכום: {str(e)}"

    def _detect_content_type_from_path(self, blob_path: str) -> str:
        """
        זיהוי סוג התוכן לפי נתיב הקובץ
        מחזיר 'video' אם הנתיב מכיל 'Videos_md' או 'document' אם מכיל 'Docs_md'
        """
        if "Videos_md" in blob_path.lower():
            return "video"
        elif "Docs_md" in blob_path.lower():
            return "document"
        else:
            # ברירת מחדל - ננסה לזהות לפי סיומת
            if blob_path.lower().endswith('.md'):
                return "document"  # ברירת מחדל למסמכים
            return "unknown"

    def _extract_section_from_path(self, blob_path: str) -> str:
        """
        חילוץ שם הסקשן מנתיב הבלוב
        לדוגמה: "Section1/Processed-data/Videos_md/file.md" -> "Section1"
        """
        path_parts = blob_path.split('/')
        for part in path_parts:
            if part.lower().startswith('section'):
                return part
        return "general"  # ברירת מחדל אם לא נמצא סקשן

    def summarize_md_file(self, blob_path: str) -> str | None:
        """
        סיכום קובץ MD מבלוב עם זיהוי אוטומטי של סוג התוכן ושמירה לבלוב

        Args:
            blob_path: נתיב לקובץ MD בבלוב

        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """
        logger.info(f"📖 Processing MD file from blob: {blob_path}")

        try:
            # זיהוי סוג התוכן מתוך ה-path
            content_type = self._detect_content_type_from_path(blob_path)
            logger.info(f"  📋 זוהה כסוג: {content_type}")

            if content_type == "unknown":
                logger.info(f"❌ לא ניתן לזהות סוג קובץ עבור: {blob_path}")
                return None

            # הורדת הקובץ מהבלוב
            temp_file_path = f"temp_{os.path.basename(blob_path)}"

            if not self.blob_manager.download_file(blob_path, temp_file_path):
                logger.info(f"❌ נכשל בהורדת הקובץ מהבלוב: {blob_path}")
                return None

            try:
                # אם זה קובץ וידאו - עבור לפארסינג מתקדם
                if content_type == "video":
                    logger.info("🎬 Video file detected - using enhanced parsing and summarization")

                    # פרסור הקובץ לחלקים השונים
                    parsed_data = self.parse_video_md_file(temp_file_path)

                    # בדיקה שיש טרנסקריפט
                    if not parsed_data.get("full_transcript"):
                        logger.info(f"❌ לא נמצא טרנסקריפט בקובץ הוידאו")
                        return None

                    # יצירת סיכום עם הפרמטרים שנפרסרו
                    summary = self.summarize_content(
                        content=parsed_data["full_transcript"],
                        content_type="video",
                        subject_type=parsed_data.get("subject_type"),
                        existing_summary=parsed_data.get("existing_summary")
                    )

                # אם זה מסמך רגיל - טיפול סטנדרטי
                else:
                    logger.info("📄 Document file - using standard processing")
                    # קריאת הקובץ
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        logger.info(f"❌ הקובץ ריק")
                        return None

                    # יצירת הסיכום
                    summary = self.summarize_content(content, content_type)

                # בדיקה שהסיכום נוצר בהצלחה
                if not summary or summary.startswith("שגיאה"):
                    logger.info(f"❌ נכשל ביצירת הסיכום")
                    return None

                # שמירת הסיכום לבלוב
                blob_summary_path = self._save_summary_to_blob(summary, blob_path)
                if blob_summary_path:
                    logger.info(f"✅ Summary saved to blob: {blob_summary_path}")
                    return blob_summary_path
                else:
                    logger.info(f"❌ נכשלה שמירת הסיכום לבלוב")
                    return None

            finally:
                # מחיקת הקובץ הזמני
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            logger.info(f"❌ שגיאה בעיבוד הקובץ: {str(e)}")
            return None

    def _save_summary_to_blob(self, summary: str, original_blob_path: str) -> str:
        """
        שמירת הסיכום לבלוב במבנה CourseID/SectionID/file_summaries/FileID.md

        Args:
            summary: הסיכום לשמירה
            original_blob_path: נתיב הקובץ המקורי בבלוב (למשל: "CS101/Section1/Docs_md/1.md")

        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """
        try:
            # פירוק הנתיב המקורי
            # למשל: "CS101/Section1/Docs_md/1.md" -> ["CS101", "Section1", "Docs_md", "1.md"]
            path_parts = original_blob_path.split('/')

            if len(path_parts) < 4:
                logger.info(f"❌ נתיב לא תקין: {original_blob_path}")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] הוא Docs_md או Videos_md
            filename = path_parts[3]  # 1.md

            # חילוץ שם הקובץ בלי סיומת
            base_name = os.path.splitext(filename)[0]  # 1

            # יצירת נתיב הסיכום החדש
            summary_blob_path = f"{course_id}/{section_id}/file_summaries/{base_name}.md"

            logger.info(f"📤 Saving summary to blob: {summary_blob_path}")

            # שמירה לבלוב
            success = self.blob_manager.upload_text_to_blob(
                text_content=summary,
                blob_name=summary_blob_path,
                container=CONTAINER_NAME
            )

            if success:
                return summary_blob_path
            else:
                logger.info(f"❌ Failed to save summary to blob")
                return None

        except Exception as e:
            logger.info(f"❌ Error saving summary to blob: {str(e)}")
            return None

    def summarize_section_from_blob(self, full_blob_path: str) -> str | None:
        """
        סיכום section שלם מכל קבצי הסיכומים ב-blob storage
        Args:
            full_blob_path: נתיב לתיקיית file_summaries (למשל: "CS101/Section1/file_summaries")
        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """

        try:
            # פירוק הנתיב: "CS101/Section1/file_summaries" -> ["CS101", "Section1", "file_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 3:
                logger.info(f"❌ נתיב לא תקין: {full_blob_path}. צריך להיות בפורמט: CourseID/SectionID/file_summaries")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] צריך להיות file_summaries

            logger.info(f"📁 CourseID: {course_id}")
            logger.info(f"📂 SectionID: {section_id}")
            logger.info(f"📂 נתיב file_summaries: {full_blob_path}")

            # יצירת BlobManager עם הקונטיינר הברירת מחדל
            blob_manager = BlobManager()

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            # סינון קבצים שנמצאים בנתיב הספציפי
            section_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not section_files:
                logger.info(f"❌ לא נמצאו קבצי סיכומים ב-{full_blob_path}")
                return None

            logger.info(f"📁 נמצאו {len(section_files)} קבצי סיכומים ב-{full_blob_path}:")
            for file in section_files:
                logger.info(f"  - {file}")

            # הורדה וקריאה של כל הקבצים ישירות לזיכרון
            all_content = ""
            successful_files = []

            for file_path in section_files:
                logger.info(f"\n📥 מוריד קובץ לזיכרון: {file_path}")

                try:
                    # הורדת הקובץ ישירות לזיכרון
                    file_bytes = blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # המרה לטקסט
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"קובץ: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f"✅ קובץ נקרא בהצלחה: {len(file_content)} תווים")
                        else:
                            logger.info(f"⚠️ קובץ ריק: {file_path}")
                    else:
                        logger.info(f"❌ נכשלה הורדת הקובץ: {file_path}")

                except Exception as e:
                    logger.info(f"❌ שגיאה בעיבוד קובץ {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f"❌ לא הצלחתי לקרוא אף קובץ מ-{full_blob_path}")
                return None

            logger.info(f"\n📊 סה\"כ עובד עם {len(successful_files)} קבצים")
            logger.info(f"📊 אורך התוכן הכולל: {len(all_content)} תווים")

            # יצירת הסיכום
            logger.info(f"\n🤖 יוצר סיכום section...")

            # הכנת prompt מיוחד לסיכום section
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

            # קריאה למודל השפה
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            section_summary = response.choices[0].message.content

            logger.info(f"✅ סיכום section נוצר בהצלחה!")
            logger.info(f"📊 אורך הסיכום: {len(section_summary)} תווים")

            # שמירת הסיכום לבלוב במבנה החדש: CourseID/section_summaries/SectionID.md
            summary_blob_path = f"{course_id}/section_summaries/{section_id}.md"

            logger.info(f"📤 שומר סיכום section ל-blob: {summary_blob_path}")

            success = blob_manager.upload_text_to_blob(
                text_content=section_summary,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f"✅ סיכום section נשמר ב-blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f"❌ נכשלה שמירת סיכום section לבלוב")
                return None

        except Exception as e:
            logger.info(f"❌ שגיאה בסיכום section: {str(e)}")
            return None

    def summarize_course_from_blob(self, full_blob_path: str) -> str | None:
        """
        סיכום קורס שלם מכל קבצי סיכומי ה-sections ב-blob storage
        Args:
            full_blob_path: נתיב לתיקיית section_summaries (למשל: "CS101/section_summaries")
        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """

        try:
            # פירוק הנתיב: "CS101/section_summaries" -> ["CS101", "section_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 2:
                logger.info(f"❌ נתיב לא תקין: {full_blob_path}. צריך להיות בפורמט: CourseID/section_summaries")
                return None

            course_id = path_parts[0]  # CS101
            # path_parts[1] צריך להיות section_summaries

            logger.info(f"📁 CourseID: {course_id}")
            logger.info(f"📂 נתיב section_summaries: {full_blob_path}")

            # יצירת BlobManager עם הקונטיינר הברירת מחדל
            blob_manager = BlobManager()

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            # סינון קבצים שנמצאים בתיקיית section_summaries
            sections_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not sections_files:
                logger.info(f"❌ לא נמצאו קבצי סיכומי sections ב-{full_blob_path}")
                return None

            logger.info(f"📁 נמצאו {len(sections_files)} קבצי סיכומי sections:")
            for file in sections_files:
                logger.info(f"  - {file}")

            # הורדה וקריאה של כל הקבצים ישירות לזיכרון
            all_content = ""
            successful_files = []

            for file_path in sections_files:
                logger.info(f"\n📥 מוריד קובץ לזיכרון: {file_path}")

                try:
                    # הורדת הקובץ ישירות לזיכרון
                    file_bytes = blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # המרה לטקסט
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"Section: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f"✅ קובץ נקרא בהצלחה: {len(file_content)} תווים")
                        else:
                            logger.info(f"⚠️ קובץ ריק: {file_path}")
                    else:
                        logger.info(f"❌ נכשלה הורדת הקובץ: {file_path}")

                except Exception as e:
                    logger.info(f"❌ שגיאה בעיבוד קובץ {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f"❌ לא הצלחתי לקרוא אף קובץ מ-{full_blob_path}")
                return None

            logger.info(f"\n📊 סה\"כ עובד עם {len(successful_files)} קבצים")
            logger.info(f"📊 אורך התוכן הכולל: {len(all_content)} תווים")

            # יצירת הסיכום
            logger.info(f"\n🤖 יוצר סיכום קורס שלם...")

            # הכנת prompt מיוחד לסיכום קורס
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

            # קריאה למודל השפה
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            course_summary = response.choices[0].message.content

            logger.info(f"✅ סיכום קורס נוצר בהצלחה!")
            logger.info(f"📊 אורך הסיכום: {len(course_summary)} תווים")

            # שמירת הסיכום לבלוב במבנה החדש: CourseID/course_summary.md
            summary_blob_path = f"{course_id}/course_summary.md"

            logger.info(f"📤 שומר סיכום קורס ל-blob: {summary_blob_path}")

            success = blob_manager.upload_text_to_blob(
                text_content=course_summary,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f"✅ סיכום קורס נשמר ב-blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f"❌ נכשלה שמירת סיכום קורס לבלוב")
                return None

        except Exception as e:
            logger.info(f"❌ שגיאה בסיכום קורס: {str(e)}")
            return None

    def save_summary_to_file(self, summary: str, original_file_path: str, output_dir: str = "summaries") -> str:
        """
        שמירת הסיכום לקובץ

        Args:
            summary: הסיכום לשמירה
            original_file_path: נתיב הקובץ המקורי
            output_dir: תיקיית הפלט

        Returns:
            נתיב הקובץ שנשמר
        """
        # יצירת תיקיית הפלט
        os.makedirs(output_dir, exist_ok=True)

        # יצירת שם קובץ לסיכום
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        summary_filename = f"{base_name}_summary.md"
        summary_path = os.path.join(output_dir, summary_filename)

        try:
            # שמירת הסיכום
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)

            logger.info(f"✅ Summary saved to: {summary_path}")
            return summary_path

        except Exception as e:
            error_msg = f"שגיאה בשמירת הסיכום: {str(e)}"
            logger.info(f"❌ {error_msg}")
            return ""


def main():
    """פונקציה ראשית לבדיקה"""
    logger.info("📝 Content Summarizer - Testing")
    logger.info("=" * 50)

    summarizer = ContentSummarizer()

    # logger.info("\n🔄 Testing summarize_md_file with blob paths...")
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
    #     logger.info(f"🔄 Testing blob: {blob_path}")
    #     logger.info(f"{'=' * 50}")
    #
    #     try:
    #         # בדיקה אם הקובץ קיים בבלוב
    #         logger.info(f"📋 Checking if blob exists...")
    #
    #         # יצירת סיכום מהבלוב
    #         summary = summarizer.summarize_md_file(blob_path)
    #
    #         if summary and not summary.startswith("שגיאה") and not summary.startswith(
    #                 "לא ניתן") and not summary.startswith("נכשל"):
    #             logger.info(f"✅ Summary created successfully!")
    #             logger.info(f"📊 Summary length: {len(summary)} characters")
    #             logger.info(f"📋 Summary preview (first 300 chars):")
    #             logger.info("-" * 40)
    #             logger.info(summary[:300] + "..." if len(summary) > 300 else summary)
    #             logger.info("-" * 40)
    #
    #             # שמירת הסיכום לקובץ מקומי
    #             summary_file = summarizer.save_summary_to_file(summary, blob_path, "summaries")
    #             if summary_file:
    #                 logger.info(f"💾 Summary saved to: {summary_file}")
    #
    #             successful_tests += 1
    #
    #         else:
    #             logger.info(f"❌ Failed to create summary: {summary}")
    #             failed_tests += 1
    #
    #     except Exception as e:
    #         logger.info(f"❌ Error processing blob {blob_path}: {str(e)}")
    #         failed_tests += 1
    #
    #     logger.info(f"\n⏱️ Waiting 2 seconds before next test...")
    #     import time
    #     time.sleep(2)


    # # בדיקת הפונקציה summarize_section_from_blob
    # logger.info("\n🔄 Testing summarize_section_from_blob...")
    #
    # # נתיב מלא בבלוב
    # full_blob_path = "CS101/Section1/file_summaries"
    #
    #
    # logger.info(f"📂 Testing full blob path: {full_blob_path}")
    #
    # try:
    #     # יצירת סיכום section
    #     result = summarizer.summarize_section_from_blob(full_blob_path)
    #
    #     if result:
    #         logger.info(f"\n✅ Section summary created successfully!")
    #         logger.info(f"📤 Summary saved to blob: {result}")
    #         logger.info(f"🎉 Test completed successfully!")
    #     else:
    #         logger.info(f"\n❌ Failed to create section summary")
    #         logger.info(f"💡 Check if there are summary files in {full_blob_path}")
    #
    # except Exception as e:
    #     logger.info(f"\n❌ Error during section summarization: {str(e)}")
    #     traceback.logger.info_exc()


    # בדיקת הפונקציה summarize_course_from_blob
    logger.info("\n🔄 Testing summarize_course_from_blob...")

    # נתיב מלא לתיקיית סיכומי ה-sections
    full_blob_path = "Discrete_mathematics/section_summaries"

    logger.info(f"📂 Testing course summary from path: {full_blob_path}")

    try:
        # יצירת סיכום קורס שלם
        result = summarizer.summarize_course_from_blob(full_blob_path)

        if result:
            logger.info(f"\n✅ Course summary created successfully!")
            logger.info(f"📤 Summary saved to blob: {result}")
            logger.info(f"🎉 Test completed successfully!")
        else:
            logger.info(f"\n❌ Failed to create course summary")
            logger.info(f"💡 Check if there are section summary files in {full_blob_path}")

    except Exception as e:
        logger.info(f"\n❌ Error during course summarization: {str(e)}")
        traceback.logger.info_exc()

    logger.info(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
