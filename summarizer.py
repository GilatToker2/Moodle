"""
Content Summarizer - מערכת סיכומים לוידאו ומסמכים
משתמשת במודל השפה מ-Azure OpenAI ליצירת סיכומים מותאמים
"""

import os
from typing import Dict, Optional
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL,
    CONTAINER_NAME
)
from blob_manager import BlobManager


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

            elif line_stripped == "## 📄 Full Transcript":
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

    def summarize_md_file(self, md_file_path: str, content_type: str = "document") -> str:
        """
        סיכום קובץ MD עם טיפול מתקדם בקבצי וידאו

        Args:
            md_file_path: נתיב לקובץ MD
            content_type: סוג התוכן - "video" או "document"

        Returns:
            הסיכום שנוצר
        """
        print(f"📖 Processing MD file: {md_file_path}")

        if not os.path.exists(md_file_path):
            error_msg = f"קובץ לא נמצא: {md_file_path}"
            print(f"❌ {error_msg}")
            return error_msg

        try:
            # אם זה קובץ וידאו - עבור לפארסינג מתקדם
            if content_type.lower() == "video":
                print("🎬 Video file detected - using enhanced parsing and summarization")

                # פרסור הקובץ לחלקים השונים
                parsed_data = self.parse_video_md_file(md_file_path)

                # בדיקה שיש טרנסקריפט
                if not parsed_data.get("full_transcript"):
                    error_msg = "לא נמצא טרנסקריפט בקובץ הוידאו"
                    print(f"❌ {error_msg}")
                    return error_msg

                # יצירת סיכום עם הפרמטרים שנפרסרו
                return self.summarize_content(
                    content=parsed_data["full_transcript"],
                    content_type="video",
                    subject_type=parsed_data.get("subject_type"),
                    existing_summary=parsed_data.get("existing_summary")
                )

            # אם זה מסמך רגיל - טיפול סטנדרטי
            else:
                print("📄 Document file - using standard processing")
                # קריאת הקובץ
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    error_msg = "הקובץ ריק"
                    print(f"❌ {error_msg}")
                    return error_msg

                # יצירת הסיכום
                summary = self.summarize_content(content, content_type)
                return summary

        except Exception as e:
            error_msg = f"שגיאה בעיבוד הקובץ: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


    def summarize_section_from_blob(self, container_name: str = None) -> str:
        """
        סיכום section שלם מכל קבצי הסיכומים ב-blob storage
        Args:
            container_name: שם הקונטיינר לעבודה (אם לא מוגדר - ישתמש בקונטיינר הדיפולטיבי)
        Returns:
            סיכום מקיף של כל הקבצים בקונטיינר
        """

        try:
            # אם לא הועבר container_name, השתמש בדיפולטיבי
            if container_name is None:
                container_name = CONTAINER_NAME
                print(f"📁 משתמש בקונטיינר הדיפולטיבי: {container_name}")
            else:
                print(f"📁 משתמש בקונטיינר שהועבר: {container_name}")

            # יצירת BlobManager עם הקונטיינר הספציפי
            blob_manager = BlobManager(container_name)

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            if not all_files:
                error_msg = f"לא נמצאו קבצי סיכומים (MD) בקונטיינר"
                print(f"❌ {error_msg}")
                return error_msg

            print(f"📁 נמצאו {len(all_files)} קבצי סיכומים:")
            for file in all_files:
                print(f"  - {file}")

            # הורדה וקריאה של כל הקבצים
            all_content = ""
            successful_files = []

            for file_path in all_files:
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
                error_msg = f"לא הצלחתי לקרוא אף קובץ מהקונטיינר"
                print(f"❌ {error_msg}")
                return error_msg

            print(f"\n📊 סה\"כ עובד עם {len(successful_files)} קבצים")
            print(f"📊 אורך התוכן הכולל: {len(all_content)} תווים")
            # print("all content:", all_content)

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

            return section_summary

        except Exception as e:
            error_msg = f"שגיאה בסיכום section: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


    def summarize_course_from_blob(self, container_name: str = None) -> str:
        """
        סיכום קורס שלם מכל קבצי הסיכומים ב-blob storage
        Args:
            container_name: שם הקונטיינר לעבודה (אם לא מוגדר - ישתמש בקונטיינר הדיפולטיבי)
        Returns:
            סיכום מקיף של כל הקורס
        """

        try:
            # אם לא הועבר container_name, השתמש בדיפולטיבי
            if container_name is None:
                container_name = CONTAINER_NAME
                print(f"📁 משתמש בקונטיינר הדיפולטיבי: {container_name}")
            else:
                print(f"📁 משתמש בקונטיינר שהועבר: {container_name}")

            # יצירת BlobManager עם הקונטיינר הספציפי
            blob_manager = BlobManager(container_name)

            # קבלת רשימת כל הקבצים בקונטיינר
            all_files = blob_manager.list_files()

            if not all_files:
                error_msg = f"לא נמצאו קבצי סיכומים (MD) בקונטיינר"
                print(f"❌ {error_msg}")
                return error_msg

            print(f"📁 נמצאו {len(all_files)} קבצי סיכומים:")
            for file in all_files:
                print(f"  - {file}")

            # הורדה וקריאה של כל הקבצים
            all_content = ""
            successful_files = []

            for file_path in all_files:
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
                error_msg = f"לא הצלחתי לקרוא אף קובץ מהקונטיינר"
                print(f"❌ {error_msg}")
                return error_msg

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

            return course_summary

        except Exception as e:
            error_msg = f"שגיאה בסיכום קורס: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

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


    # בדיקת סיכום section מ-blob storage
    print("\n🔄 Testing Section Summary...")
    section_summary = summarizer.summarize_section_from_blob()
    if section_summary and not section_summary.startswith("שגיאה") and not section_summary.startswith("לא נמצאו"):
        summarizer.save_summary_to_file(section_summary, "container_section", "summaries")
        print("✅ Section summary completed successfully!")
    else:
        print(f"❌ Section summary failed: {section_summary}")

    # בדיקת סיכום קורס שלם מ-blob storage
    print("\n🔄 Testing Course Summary...")
    course_summary = summarizer.summarize_course_from_blob()
    if course_summary and not course_summary.startswith("שגיאה") and not course_summary.startswith("לא נמצאו"):
        summarizer.save_summary_to_file(course_summary, "full_course", "summaries")
        print("✅ Course summary completed successfully!")
    else:
        print(f"❌ Course summary failed: {course_summary}")

    print(f"\n🎉 Testing completed!")

    #
    # # רשימת קבצים לבדיקה
    # test_files = [
    #     # ("docs_md/Ex5Sol.md", "document"),
    #     # ("docs_md/bdida_tirgul_02.md", "document"),
    #     ("docs_md/DL_14_LLMs.md", "document"),
    #     ("videos_md/L2_d1847b82963a0ef0fc97d72ef5602cf785490bf1_merged_30s.md", "video"),
    #     # ("videos_md/L9_18f0d24bb7e45223abf842cdc1274de65fc7d620 - Trim.md", "video")
    # ]
    #
    # for file_path, content_type in test_files:
    #     if os.path.exists(file_path):
    #         print(f"\n🔄 Processing: {os.path.basename(file_path)} ({content_type})")
    #         print("-" * 60)
    #
    #         try:
    #             # יצירת סיכום
    #             summary = summarizer.summarize_md_file(file_path, content_type)
    #
    #             if summary.startswith("שגיאה"):
    #                 print(f"❌ Error: {summary}")
    #                 continue
    #
    #             print(f"✅ Summary created successfully!")
    #             print(f"📋 Summary preview (first 200 chars):")
    #             print(summary[:200] + "..." if len(summary) > 200 else summary)
    #
    #             # שמירת הסיכום
    #             summary_file = summarizer.save_summary_to_file(summary, file_path)
    #             print(f"💾 Summary saved to: {summary_file}")
    #
    #         except Exception as e:
    #             print(f"❌ Error processing file: {str(e)}")
    #     else:
    #         print(f"❌ File not found: {file_path}")

    print(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
