"""
Content Summarizer - מערכת סיכומים לוידאו ומסמכים
משתמשת במודל השפה מ-Azure OpenAI ליצירת סיכומים מותאמים
"""

import os
import traceback
from typing import Dict
from openai import AzureOpenAI
from Config.Config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL,
    CONTAINER_NAME
)
from Source.Services.Blob_manager import BlobManager


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
        print(f"✅ ContentSummarizer initialized with model: {self.model_name}")

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
        """הכנת prompt לסיכום קורס שלם"""
        return """אתה מומחה לסיכום חומרי לימוד אקדמיים. קיבלת אוסף של סיכומי Section מתוך קורס אוניברסיטאי שלם.
    כל סיכום Section מייצג חלק משמעותי מהחומר שכבר סוכם בצורה מפורטת. המטרה שלך עכשיו היא לאחד את כל סיכומי ה-Sections לכדי סיכום-על **מפורט**, מקיף ופדגוגי של הקורס כולו.

    זכור: המטרה שלך **אינה לקצר או להשמיט פרטים** אלא לארגן, לשלב ולהציג מחדש את כל המידע הקיים כך שהסטודנט יוכל ללמוד את כל תוכן הקורס **מתוך הסיכום הכולל בלבד** — גם מבלי לקרוא את ה-Sections המקוריים.
    השתמש בכל המידע הקיים — כולל דוגמאות, הסברים, הגדרות והערות — והצג אותו כחלק ממסע לימודי שלם.

    המטרה שלך:
    - ליצור סיכום שלם של הקורס כולו, המכסה את כל ה-Sections שקיבלת.
    - לזהות את המבנה הלוגי וההתפתחות הפדגוגית של הקורס.
    - לארגן את החומר כך שידגיש את ההתקדמות מהבסיסי למתקדם.
    - להראות את הקשרים והרצף בין הנושאים בכל ה-Sections.

    המשימה שלך:
    - עבור על כל סיכומי ה-Sections והבין מהם הנושאים המרכזיים של הקורס.
    - מצא את החוט המקשר בין ה-Sections — איך רעיונות מתפתחים לאורך הקורס.
    - סדר את התוכן לפי התקדמות לוגית וברורה.
    - שמור על עומק ורוחב החומר המקורי — אל תחסוך בפרטים.
    - הדגש מושגים מרכזיים שחוזרים ומעמיקים בין ה-Sections.

    מבנה הפלט:
    1. **פתיחה כללית** — משפט או שניים שמסבירים מה מטרת הקורס ומה הסטודנט ירוויח מהסיכום הכולל.
    2. **נושאים מרכזיים** — רשימה תמציתית שמראה מהם התחומים והנושאים החשובים שעוברים כחוט שני לאורך כל הקורס.
    3. **סיכום מפורט לפי נושאים** — שילוב של כל המידע הקיים, עם הסברים, דוגמאות והערות כפי שנכללו בסיכומי ה-Sections.
    4. **מושגי מפתח והתקדמות הידע + המלצות ללמידה** — הסבר איך הנושאים מתפתחים ונבנים לאורך הקורס והמלצות פרקטיות לחזרה ותרגול.

    זכור:
    - אל תדלג על פרטים — הסיכום צריך להיות שלם ומעמיק.
    - שמור על טון ברור ומנחה, כאילו אתה מלווה את הסטודנט בכל שלב.
    - סדר הכל כך שישקף את הזרימה הפדגוגית וההתפתחות ההדרגתית של הקורס.

    סיכום הקורס:
    """

    def parse_video_md_file(self, md_file_path: str) -> Dict:
        """
        פירוק קובץ video.md לחלקים הספציפיים שלו

        Args:
            md_file_path: נתיב לקובץ video.md

        Returns:
            מילון עם החלקים השונים של הקובץ
        """
        print(f"📖 Parsing video MD file: {md_file_path}")

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

        # print(f"🔍 Detected subject type: {subject_type}")
        # print(f"📝 Existing summary length: {len(existing_summary) if existing_summary else 0} chars")
        # print(f"📄 Full transcript length: {len(full_transcript) if full_transcript else 0} chars")
        #
        # # הדפסת תוכן מפורט
        # print("\n" + "=" * 60)
        # print("📋 PARSED CONTENT DETAILS:")
        # print("=" * 60)
        #
        # print(f"\n🎓 SUBJECT TYPE:")
        # if subject_type:
        #     print(f"'{subject_type}'")
        # else:
        #     print("None")
        #
        # print(f"\n📝 EXISTING SUMMARY:")
        # if existing_summary:
        #     print(f"'{existing_summary}'")
        # else:
        #     print("None")

        # print(f"\n📄 TRANSCRIPT PREVIEW:")
        # if full_transcript:
        #     lines = full_transcript.split('\n')
        #     if len(lines) > 4:
        #         print("First 2 lines:")
        #         for i, line in enumerate(lines[:2]):
        #             print(f"  {i + 1}: {line}")
        #         print("  ...")
        #         print("Last 2 lines:")
        #         for i, line in enumerate(lines[-2:], len(lines) - 1):
        #             print(f"  {i}: {line}")
        #     else:
        #         print("Full transcript (short):")
        #         for i, line in enumerate(lines):
        #             print(f"  {i + 1}: {line}")
        # else:
        #     print("None")
        #
        # print("=" * 60)

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
        print(f"📝 Creating summary for {content_type} content...")
        print(f"📊 Content length: {len(content)} characters")

        try:
            # בחירת prompt לפי סוג התוכן
            if content_type.lower() == "video":
                print(f"🎓 Subject type: {subject_type}")
                print(f"📝 Has existing summary: {bool(existing_summary)}")
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
            print(f"🤖 Calling {self.model_name} for summarization...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=32768,
                temperature=0.3,  # יציבות בסיכום
                top_p=0.8
            )

            summary = response.choices[0].message.content

            print(f"✅ Summary created successfully!")
            print(f"📊 Summary length: {len(summary)} characters")

            return summary

        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return f"שגיאה ביצירת סיכום: {str(e)}"

    def _detect_content_type_from_path(self, blob_path: str) -> str:
        """
        זיהוי סוג התוכן לפי נתיב הקובץ
        מחזיר 'video' אם הנתיב מכיל 'videos_md' או 'document' אם מכיל 'docs_md'
        """
        if "videos_md" in blob_path.lower():
            return "video"
        elif "docs_md" in blob_path.lower():
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
        print(f"📖 Processing MD file from blob: {blob_path}")

        try:
            # זיהוי סוג התוכן מתוך ה-path
            content_type = self._detect_content_type_from_path(blob_path)
            print(f"  📋 זוהה כסוג: {content_type}")

            if content_type == "unknown":
                print(f"❌ לא ניתן לזהות סוג קובץ עבור: {blob_path}")
                return None

            # הורדת הקובץ מהבלוב
            temp_file_path = f"temp_{os.path.basename(blob_path)}"

            if not self.blob_manager.download_file(blob_path, temp_file_path):
                print(f"❌ נכשל בהורדת הקובץ מהבלוב: {blob_path}")
                return None

            try:
                # אם זה קובץ וידאו - עבור לפארסינג מתקדם
                if content_type == "video":
                    print("🎬 Video file detected - using enhanced parsing and summarization")

                    # פרסור הקובץ לחלקים השונים
                    parsed_data = self.parse_video_md_file(temp_file_path)

                    # בדיקה שיש טרנסקריפט
                    if not parsed_data.get("full_transcript"):
                        print(f"❌ לא נמצא טרנסקריפט בקובץ הוידאו")
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
                    print("📄 Document file - using standard processing")
                    # קריאת הקובץ
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        print(f"❌ הקובץ ריק")
                        return None

                    # יצירת הסיכום
                    summary = self.summarize_content(content, content_type)

                # בדיקה שהסיכום נוצר בהצלחה
                if not summary or summary.startswith("שגיאה"):
                    print(f"❌ נכשל ביצירת הסיכום")
                    return None

                # שמירת הסיכום לבלוב
                blob_summary_path = self._save_summary_to_blob(summary, blob_path)
                if blob_summary_path:
                    print(f"✅ Summary saved to blob: {blob_summary_path}")
                    return blob_summary_path
                else:
                    print(f"❌ נכשלה שמירת הסיכום לבלוב")
                    return None

            finally:
                # מחיקת הקובץ הזמני
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            print(f"❌ שגיאה בעיבוד הקובץ: {str(e)}")
            return None

    def _save_summary_to_blob(self, summary: str, original_blob_path: str) -> str:
        """
        שמירת הסיכום לבלוב במבנה Summaries/Section/filename.md

        Args:
            summary: הסיכום לשמירה
            original_blob_path: נתיב הקובץ המקורי בבלוב

        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """
        try:
            # חילוץ שם הסקשן מהנתיב המקורי
            section_name = self._extract_section_from_path(original_blob_path)

            # חילוץ שם הקובץ
            filename = os.path.basename(original_blob_path)
            base_name = os.path.splitext(filename)[0]
            summary_filename = f"{base_name}_summary.md"

            # יצירת נתיב הסיכום החדש
            summary_blob_path = f"Summaries/{section_name}/{summary_filename}"

            print(f"📤 Saving summary to blob: {summary_blob_path}")

            # שמירה לבלוב
            success = self.blob_manager.upload_text_to_blob(
                text_content=summary,
                blob_name=summary_blob_path,
                container=CONTAINER_NAME
            )

            if success:
                return summary_blob_path
            else:
                print(f"❌ Failed to save summary to blob")
                return None

        except Exception as e:
            print(f"❌ Error saving summary to blob: {str(e)}")
            return None

    def summarize_section_from_blob(self, full_blob_path: str) -> str | None:
        """
        סיכום section שלם מכל קבצי הסיכומים ב-blob storage
        Args:
            full_blob_path: נתיב מלא בבלוב (למשל: "course1/Summaries/Section1")
        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """

        try:
            # פיצול הנתיב המלא לקונטיינר ונתיב פנימי
            path_parts = full_blob_path.split('/', 1)
            if len(path_parts) < 2:
                print(f"❌ נתיב לא תקין: {full_blob_path}. צריך להיות בפורמט: container/path")
                return None

            container_name = path_parts[0]
            section_path = path_parts[1]

            print(f"📁 קונטיינר: {container_name}")
            print(f"📂 נתיב section: {section_path}")

            # יצירת BlobManager עם הקונטיינר הספציפי
            blob_manager = BlobManager(container_name)

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            # סינון קבצים שנמצאים בנתיב הספציפי
            section_files = [f for f in all_files if f.startswith(section_path + "/") and f.endswith(".md")]

            if not section_files:
                print(f"❌ לא נמצאו קבצי סיכומים ב-{section_path}")
                return None

            print(f"📁 נמצאו {len(section_files)} קבצי סיכומים ב-{section_path}:")
            for file in section_files:
                print(f"  - {file}")

            # הורדה וקריאה של כל הקבצים
            all_content = ""
            successful_files = []

            for file_path in section_files:
                print(f"\n📥 מוריד קובץ: {file_path}")

                # יצירת נתיב זמני לקובץ
                temp_file_path = f"temp_{os.path.basename(file_path)}"

                try:
                    # הורדת הקובץ
                    if blob_manager.download_file(file_path, temp_file_path):
                        # קריאת התוכן
                        with open(temp_file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"קובץ: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            print(f"✅ קובץ נקרא בהצלחה: {len(file_content)} תווים")
                        else:
                            print(f"⚠️ קובץ ריק: {file_path}")

                        # מחיקת הקובץ הזמני
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

                except Exception as e:
                    print(f"❌ שגיאה בעיבוד קובץ {file_path}: {e}")
                    # מחיקת הקובץ הזמני במקרה של שגיאה
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    continue

            if not successful_files:
                print(f"❌ לא הצלחתי לקרוא אף קובץ מ-{section_path}")
                return None

            print(f"\n📊 סה\"כ עובד עם {len(successful_files)} קבצים")
            print(f"📊 אורך התוכן הכולל: {len(all_content)} תווים")

            # יצירת הסיכום
            print(f"\n🤖 יוצר סיכום section...")

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
                max_tokens=32768,
                temperature=0.3,
                top_p=0.8
            )

            section_summary = response.choices[0].message.content

            print(f"✅ סיכום section נוצר בהצלחה!")
            print(f"📊 אורך הסיכום: {len(section_summary)} תווים")

            # שמירת הסיכום לבלוב
            section_name = os.path.basename(section_path)  # חילוץ שם הסקשן מהנתיב
            summary_filename = f"{section_name}_section_summary.md"
            summary_blob_path = f"Summaries/Sections_summary/{summary_filename}"

            print(f"📤 שומר סיכום section ל-blob: {summary_blob_path}")

            success = blob_manager.upload_text_to_blob(
                text_content=section_summary,
                blob_name=summary_blob_path,
                container=container_name
            )

            if success:
                print(f"✅ סיכום section נשמר ב-blob: {summary_blob_path}")
                return summary_blob_path
            else:
                print(f"❌ נכשלה שמירת סיכום section לבלוב")
                return None

        except Exception as e:
            print(f"❌ שגיאה בסיכום section: {str(e)}")
            return None

    def summarize_course_from_blob(self, full_blob_path: str) -> str | None:
        """
        סיכום קורס שלם מכל קבצי סיכומי ה-sections ב-blob storage
        Args:
            full_blob_path: נתיב מלא לתיקיית סיכומי ה-sections (למשל: "course1/Summaries/Sections_summary")
        Returns:
            נתיב הסיכום בבלוב או None אם נכשל
        """

        try:
            # פיצול הנתיב המלא לקונטיינר ונתיב פנימי
            path_parts = full_blob_path.split('/', 1)
            if len(path_parts) < 2:
                print(f"❌ נתיב לא תקין: {full_blob_path}. צריך להיות בפורמט: container/path")
                return None

            container_name = path_parts[0]
            sections_path = path_parts[1]

            print(f"📁 קונטיינר: {container_name}")
            print(f"📂 נתיב sections: {sections_path}")

            # יצירת BlobManager עם הקונטיינר הספציפי
            blob_manager = BlobManager(container_name)

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            # סינון קבצים שנמצאים בתיקיית סיכומי ה-sections
            sections_files = [f for f in all_files if f.startswith(sections_path + "/") and f.endswith(".md")]

            if not sections_files:
                print(f"❌ לא נמצאו קבצי סיכומי sections ב-{sections_path}")
                return None

            print(f"📁 נמצאו {len(sections_files)} קבצי סיכומי sections:")
            for file in sections_files:
                print(f"  - {file}")

            # הורדה וקריאה של כל הקבצים ישירות לזיכרון
            all_content = ""
            successful_files = []

            for file_path in sections_files:
                print(f"\n📥 מוריד קובץ לזיכרון: {file_path}")

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
                            print(f"✅ קובץ נקרא בהצלחה: {len(file_content)} תווים")
                        else:
                            print(f"⚠️ קובץ ריק: {file_path}")
                    else:
                        print(f"❌ נכשלה הורדת הקובץ: {file_path}")

                except Exception as e:
                    print(f"❌ שגיאה בעיבוד קובץ {file_path}: {e}")
                    continue

            if not successful_files:
                print(f"❌ לא הצלחתי לקרוא אף קובץ מ-{sections_path}")
                return None

            print(f"\n📊 סה\"כ עובד עם {len(successful_files)} קבצים")
            print(f"📊 אורך התוכן הכולל: {len(all_content)} תווים")

            # יצירת הסיכום
            print(f"\n🤖 יוצר סיכום קורס שלם...")

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
                max_tokens=32768,
                temperature=0.3,
                top_p=0.8
            )

            course_summary = response.choices[0].message.content

            print(f"✅ סיכום קורס נוצר בהצלחה!")
            print(f"📊 אורך הסיכום: {len(course_summary)} תווים")

            # שמירת הסיכום לבלוב
            course_summary_filename = "course_summary.md"
            summary_blob_path = f"Summaries/{course_summary_filename}"

            print(f"📤 שומר סיכום קורס ל-blob: {summary_blob_path}")

            success = blob_manager.upload_text_to_blob(
                text_content=course_summary,
                blob_name=summary_blob_path,
                container=container_name
            )

            if success:
                print(f"✅ סיכום קורס נשמר ב-blob: {summary_blob_path}")
                return summary_blob_path
            else:
                print(f"❌ נכשלה שמירת סיכום קורס לבלוב")
                return None

        except Exception as e:
            print(f"❌ שגיאה בסיכום קורס: {str(e)}")
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

            print(f"✅ Summary saved to: {summary_path}")
            return summary_path

        except Exception as e:
            error_msg = f"שגיאה בשמירת הסיכום: {str(e)}"
            print(f"❌ {error_msg}")
            return ""


def main():
    """פונקציה ראשית לבדיקה"""
    print("📝 Content Summarizer - Testing")
    print("=" * 50)

    summarizer = ContentSummarizer()

    # print("\n🔄 Testing summarize_md_file with blob paths...")
    #
    # test_blob_paths = [
    #     "Section1/Processed-data/Videos_md/L1_091004f349688522f773afc884451c9af6da18fb_Trim.md",
    #     # "Section1/Processed-data/Docs_md/bdida_tirgul_02.md",
    #     # "Section1/Processed-data/Docs_md/Ex5Sol.md"
    # ]
    #
    # successful_tests = 0
    # failed_tests = 0
    #
    # for blob_path in test_blob_paths:
    #     print(f"\n{'=' * 50}")
    #     print(f"🔄 Testing blob: {blob_path}")
    #     print(f"{'=' * 50}")
    #
    #     try:
    #         # בדיקה אם הקובץ קיים בבלוב
    #         print(f"📋 Checking if blob exists...")
    #
    #         # יצירת סיכום מהבלוב
    #         summary = summarizer.summarize_md_file(blob_path)
    #
    #         if summary and not summary.startswith("שגיאה") and not summary.startswith(
    #                 "לא ניתן") and not summary.startswith("נכשל"):
    #             print(f"✅ Summary created successfully!")
    #             print(f"📊 Summary length: {len(summary)} characters")
    #             print(f"📋 Summary preview (first 300 chars):")
    #             print("-" * 40)
    #             print(summary[:300] + "..." if len(summary) > 300 else summary)
    #             print("-" * 40)
    #
    #             # שמירת הסיכום לקובץ מקומי
    #             summary_file = summarizer.save_summary_to_file(summary, blob_path, "summaries")
    #             if summary_file:
    #                 print(f"💾 Summary saved to: {summary_file}")
    #
    #             successful_tests += 1
    #
    #         else:
    #             print(f"❌ Failed to create summary: {summary}")
    #             failed_tests += 1
    #
    #     except Exception as e:
    #         print(f"❌ Error processing blob {blob_path}: {str(e)}")
    #         failed_tests += 1
    #
    #     print(f"\n⏱️ Waiting 2 seconds before next test...")
    #     import time
    #     time.sleep(2)


    # # בדיקת הפונקציה summarize_section_from_blob
    # print("\n🔄 Testing summarize_section_from_blob...")
    #
    # # נתיב מלא בבלוב
    # full_blob_path = "course1/Summaries/Section1"
    #
    # print(f"📂 Testing full blob path: {full_blob_path}")
    #
    # try:
    #     # יצירת סיכום section
    #     result = summarizer.summarize_section_from_blob(full_blob_path)
    #
    #     if result:
    #         print(f"\n✅ Section summary created successfully!")
    #         print(f"📤 Summary saved to blob: {result}")
    #         print(f"🎉 Test completed successfully!")
    #     else:
    #         print(f"\n❌ Failed to create section summary")
    #         print(f"💡 Check if there are summary files in {full_blob_path}")
    #
    # except Exception as e:
    #     print(f"\n❌ Error during section summarization: {str(e)}")
    #     traceback.print_exc()


    # בדיקת הפונקציה summarize_course_from_blob
    print("\n🔄 Testing summarize_course_from_blob...")

    # נתיב מלא לתיקיית סיכומי ה-sections
    full_blob_path = "course1/Summaries/Sections_summary"

    print(f"📂 Testing course summary from path: {full_blob_path}")

    try:
        # יצירת סיכום קורס שלם
        result = summarizer.summarize_course_from_blob(full_blob_path)

        if result:
            print(f"\n✅ Course summary created successfully!")
            print(f"📤 Summary saved to blob: {result}")
            print(f"🎉 Test completed successfully!")
        else:
            print(f"\n❌ Failed to create course summary")
            print(f"💡 Check if there are section summary files in {full_blob_path}")

    except Exception as e:
        print(f"\n❌ Error during course summarization: {str(e)}")
        traceback.print_exc()

    print(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
