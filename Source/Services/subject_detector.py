"""
Subject Detection Service
מזהה סוג מקצוע (מתמטי/הומני) על בסיס רשימת קבצים וסרטונים
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


class SubjectDetector:
    """מזהה סוג מקצוע על בסיס ניתוח קבצים וסרטונים"""

    def __init__(self, max_vid: int = 5, max_doc: int = 5):
        self.blob_manager = BlobManager(container_name="processeddata")
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.max_vid = max_vid
        self.max_doc = max_doc
        print(f"🔧 מגבלות קבצים: מקסימום {self.max_vid} וידאו, {self.max_doc} מסמכים")

    def extract_subject_type_from_video_md(self, md_content: str) -> Optional[str]:
        """
        חילוץ סוג המקצוע מקובץ markdown של וידאו

        Args:
            md_content: תוכן קובץ ה-markdown

        Returns:
            "מתמטי" או "הומני" או None אם לא נמצא
        """
        print("🎬 מתחיל חילוץ סוג מקצוע מקובץ וידאו markdown")
        # חיפוש אחר הסקציה "סוג מקצוע"
        pattern = r'## 🎓 סוג מקצוע\s*\n\s*([^\n]+)'
        match = re.search(pattern, md_content)

        if match:
            subject_type = match.group(1).strip()
            print(f"✅ נמצא סוג מקצוע: {subject_type}")
            if subject_type in ['מתמטי', 'הומני']:
                return subject_type

        print("⚠️ לא נמצא סוג מקצוע בקובץ הוידאו")
        return None

    def extract_full_transcript_from_video_md(self, md_content: str) -> str:
        """
        חילוץ הטרנסקריפט המלא מקובץ markdown של וידאו

        Args:
            md_content: תוכן קובץ ה-markdown

        Returns:
            הטרנסקריפט המלא או מחרוזת ריקה
        """
        print("📄 מתחיל חילוץ טרנסקריפט מלא מקובץ וידאו")
        # חיפוש אחר הסקציה "טרנסקריפט מלא"
        pattern = r'## 📄 טרנסקריפט מלא\s*\n(.*?)(?=\n## |\n$)'
        match = re.search(pattern, md_content, re.DOTALL)

        if match:
            transcript = match.group(1).strip()
            print(f"✅ נמצא טרנסקריפט באורך {len(transcript)} תווים")
            return transcript

        print("⚠️ לא נמצא טרנסקריפט בקובץ הוידאו")
        return ""

    def analyze_files_with_llm(self, file_contents: List[Dict[str, str]]) -> str:
        """
        ניתוח קבצים באמצעות מודל שפה לקביעת סוג המקצוע
        מגביל מספר קבצים לפי max_vid ו-max_doc

        Args:
            file_contents: רשימת מילונים עם 'path' ו-'content' של כל קובץ

        Returns:
            "מתמטי" או "הומני"
        """
        print(f"🤖 מתחיל ניתוח עם מודל השפה עבור {len(file_contents)} קבצים")
        print(f"📏 מגבלות: מקסימום {self.max_vid} וידאו, {self.max_doc} מסמכים")

        # הפרדת קבצים לוידאו ומסמכים
        video_files = []
        doc_files = []

        for file_info in file_contents:
            if '/Videos_md/' in file_info['path']:
                video_files.append(file_info)
            else:
                doc_files.append(file_info)

        print(f"  📊 נמצאו: {len(video_files)} וידאו, {len(doc_files)} מסמכים")

        # הגבלת מספר הקבצים
        selected_videos = video_files[:self.max_vid]
        selected_docs = doc_files[:self.max_doc]

        if len(video_files) > self.max_vid:
            print(f"  ⚠️ הגבלתי וידאו ל-{self.max_vid} מתוך {len(video_files)}")
        if len(doc_files) > self.max_doc:
            print(f"  ⚠️ הגבלתי מסמכים ל-{self.max_doc} מתוך {len(doc_files)}")

        # שילוב הקבצים הנבחרים
        selected_files = selected_videos + selected_docs

        print(f"  ✅ מנתח {len(selected_files)} קבצים: {len(selected_videos)} וידאו + {len(selected_docs)} מסמכים")

        # יצירת פרומפט עם הקבצים הנבחרים
        prompt = """אתה מומחה בסיווג תוכן אקדמי. עליך לנתח את התוכן הבא ולקבוע האם זה מקצוע מתמטי/טכני או הומני.

קריטריונים לסיווג:
- מתמטי: מתמטיקה, פיזיקה, מדעי המחשב, הנדסה, סטטיסטיקה, לוגיקה, אלגוריתמים
- הומני: ספרות, היסטוריה, פילוסופיה, פסיכולוגיה, סוציולוגיה, אמנות, שפות

תוכן הקבצים לניתוח:

"""

        for i, file_info in enumerate(selected_files, 1):
            print(f"  📄 מוסיף לניתוח קובץ {i}: {file_info['path']}")
            prompt += f"\n--- קובץ {i}: {file_info['path']} ---\n"

            # אם זה וידאו, השתמש בטרנסקריפט המלא בלבד
            if '/Videos_md/' in file_info['path']:
                transcript = self.extract_full_transcript_from_video_md(file_info['content'])
                content = transcript if transcript else file_info['content']
                print(f"    🎬 וידאו - אורך טרנסקריפט: {len(content)} תווים")
            else:
                # עבור מסמכים, קח את כל התוכן
                content = file_info['content']
                print(f"    📋 מסמך - אורך תוכן: {len(content)} תווים")

            prompt += content
            prompt += "\n" + "="*50 + "\n"

        prompt += """
על בסיס התוכן שניתח, השב במילה אחת בלבד:
- "מתמטי" אם זה מקצוע מתמטי/טכני
- "הומני" אם זה מקצוע הומני

תשובה:"""

        print(f"  📊 אורך פרומפט כולל: {len(prompt)} תווים")
        print(f"  🔄 שולח בקשה למודל השפה...")

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
            print(f"  🎯 תשובת המודל: '{result}'")

            # וידוא שהתשובה תקינה
            if result in ['מתמטי', 'הומני']:
                print(f"  ✅ תשובה תקינה: {result}")
                return result
            else:
                print(f"  ⚠️ תשובה לא צפויה מהמודל: {result}, מחזיר 'לא זוהה'")
                return "לא זוהה"

        except Exception as e:
            print(f"  ❌ שגיאה בניתוח עם מודל השפה: {e}")
            return "לא זוהה"

    def detect_subject_from_course_path(self, course_path: str) -> str:
        """
        זיהוי סוג המקצוע על בסיס path לblob של קורס

        הלוגיקה:
        1. מוצא את כל קבצי הוידאו והמסמכים בקורס
        2. אם יש יותר מ-2 קבצי וידאו עם subject_type זהה - מחזיר אותו
        3. אחרת - מעביר למודל שפה לניתוח של כל הקבצים

        Args:
            course_path: נתיב לקורס ב-blob storage (למשל: "CS101")

        Returns:
            "מתמטי", "הומני", או "לא זוהה"
        """
        print(f"🎓 מתחיל זיהוי סוג מקצוע עבור קורס: {course_path}")

        # מציאת כל הקבצים בקורס
        all_files = self.blob_manager.list_files()

        # סינון קבצים שמתחילים ב-course_path ומסתיימים ב-.md
        course_files = [
            f for f in all_files
            if f.startswith(course_path + "/") and f.endswith(".md") and
            ("/Videos_md/" in f or "/Docs_md/" in f)
        ]

        if not course_files:
            print(f"  ❌ לא נמצאו קבצים בקורס: {course_path}")
            return "לא זוהה"

        print(f"  📁 נמצאו {len(course_files)} קבצים בקורס:")
        for file in course_files:
            print(f"    - {file}")

        # קריאה לפונקציה הקיימת עם רשימת הקבצים
        return self.detect_subject(course_files)

    def detect_subject(self, file_paths: List[str]) -> str:
        """
        זיהוי סוג המקצוע על בסיס רשימת קבצים

        הלוגיקה:
        1. אם כל הוידאו עם subject_type זהה - מחזיר אותו
        2. אחרת - מעביר למודל שפה לניתוח

        Args:
            file_paths: רשימת נתיבי קבצים ב-blob storage

        Returns:
            "מתמטי", "הומני", או "לא זוהה"
        """
        print(f"🔍 מזהה סוג מקצוע עבור {len(file_paths)} קבצים")

        video_subject_types = []
        file_contents = []

        # עיבוד כל קובץ
        for file_path in file_paths:
            print(f"  📄 מעבד קובץ: {file_path}")

            # הורדת תוכן הקובץ
            content = self.blob_manager.download_to_memory(file_path)
            if not content:
                print(f"    ⚠️ לא ניתן להוריד קובץ: {file_path}")
                continue

            try:
                md_content = content.decode('utf-8')
                file_contents.append({
                    'path': file_path,
                    'content': md_content
                })

                # אם זה קובץ וידאו, נסה לחלץ subject_type
                if '/Videos_md/' in file_path:
                    subject_type = self.extract_subject_type_from_video_md(md_content)
                    if subject_type:
                        video_subject_types.append(subject_type)
                        print(f"    ✅ נמצא סוג מקצוע: {subject_type}")
                    else:
                        print(f"    ⚠️ לא נמצא סוג מקצוע בוידאו")

            except UnicodeDecodeError:
                print(f"    ❌ שגיאה בקריאת קובץ: {file_path}")
                continue

        # בדיקה אם כל הוידאו עם אותו subject_type (רק אם יש לפחות 2 וידאו)
        if len(video_subject_types) >= 2:
            unique_types = list(set(video_subject_types))
            print(f"  📊 נמצאו סוגי מקצוע בוידאו: {video_subject_types}")

            if len(unique_types) == 1:
                result = unique_types[0]
                print(f"  ✅ כל הוידאו ({len(video_subject_types)}) עם אותו סוג מקצוע: {result}")
                return result
            else:
                print(f"  🔄 נמצאו סוגי מקצוע שונים בוידאו, מעביר למודל שפה")
        elif len(video_subject_types) == 1:
            print(f"  ⚠️ נמצא רק וידאו אחד עם סוג מקצוע: {video_subject_types[0]}, מעביר למודל שפה לאימות")
        else:
            print(f"  🔄 לא נמצאו וידאו עם סוג מקצוע מוגדר, מעביר למודל שפה")

        # אם אין הסכמה או אין וידאו עם subject_type, השתמש במודל שפה
        if not file_contents:
            print("  ❌ אין תוכן לניתוח")
            return "לא זוהה"

        print(f"  🤖 מנתח {len(file_contents)} קבצים עם מודל שפה...")
        result = self.analyze_files_with_llm(file_contents)
        print(f"  ✅ תוצאת ניתוח מודל השפה: {result}")

        return result


def detect_subject_from_paths(file_paths: List[str], max_vid: int = 5, max_doc: int = 5) -> str:
    """
    פונקציה נוחה לזיהוי סוג מקצוע מרשימת קבצים

    Args:
        file_paths: רשימת נתיבי קבצים ב-blob storage
        max_vid: מספר מקסימלי של קבצי וידאו לניתוח
        max_doc: מספר מקסימלי של קבצי מסמכים לניתוח

    Returns:
        "מתמטי", "הומני", או "לא זוהה"
    """
    print(f"🚀 מתחיל זיהוי סוג מקצוע עבור {len(file_paths)} קבצים")
    detector = SubjectDetector(max_vid=max_vid, max_doc=max_doc)
    return detector.detect_subject(file_paths)


def detect_subject_from_course(course_path: str, max_vid: int = 5, max_doc: int = 5) -> str:
    """
    פונקציה נוחה לזיהוי סוג מקצוע מ-path של קורס

    Args:
        course_path: נתיב לקורס ב-blob storage (למשל: "CS101")
        max_vid: מספר מקסימלי של קבצי וידאו לניתוח
        max_doc: מספר מקסימלי של קבצי מסמכים לניתוח

    Returns:
        "מתמטי", "הומני", או "לא זוהה"
    """
    print(f"🎯 מתחיל זיהוי סוג מקצוע עבור קורס: {course_path}")
    detector = SubjectDetector(max_vid=max_vid, max_doc=max_doc)
    return detector.detect_subject_from_course_path(course_path)


if __name__ == "__main__":
    # בדיקה של הפונקציה
    print("🧪 בדיקת זיהוי סוג מקצוע")
    print("=" * 50)

    # בדיקה 1: זיהוי מקורס שלם (הפונקציה החדשה)
    print("\n🔍 בדיקה 1: זיהוי סוג מקצוע מקורס שלם")
    print("-" * 40)

    course_path = "CS101"

    try:
        result = detect_subject_from_course(course_path)
        print(f"🎯 תוצאה עבור קורס {course_path}: {result}")
    except Exception as e:
        print(f"❌ שגיאה בבדיקת קורס: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ בדיקות הושלמו!")
