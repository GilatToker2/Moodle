import requests
import time
import os
import subprocess
from datetime import datetime
from typing import Optional, Dict, List
from Config.config import VIDEO_INDEXER_ACCOUNT_ID, VIDEO_INDEXER_LOCATION, VIDEO_INDEXER_TOKEN
from Source.Services.blob_manager import BlobManager



class VideoIndexerManager:
    """
    מנהל עיבוד וידאו באמצעות Azure Video Indexer
    מתמחה בעיבוד וידאו מ-blob storage ויצירת קבצי markdown
    """

    def __init__(self):
        self.token = VIDEO_INDEXER_TOKEN
        self.account_id = VIDEO_INDEXER_ACCOUNT_ID
        self.location = VIDEO_INDEXER_LOCATION
        self.supported_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']

    def _get_params_with_token(self, additional_params=None):
        """קבלת פרמטרים עם טוקן גישה."""
        params = {"accessToken": self.token}
        if additional_params:
            params.update(additional_params)
        return params

    def upload_video_from_url(self, video_sas_url: str, video_name: str) -> str:
        """
        העלאת וידאו ל-Video Indexer באמצעות SAS URL

        Args:
            video_sas_url: SAS URL של הוידאו ב-blob storage
            video_name: שם הוידאו ב-Video Indexer
        """
        print(f"📤 מעלה וידאו: {video_name}")

        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos"

        params = self._get_params_with_token({
            "name": video_name,
            "privacy": "Private",
            "videoUrl": video_sas_url,
            "language": "he-IL",
            "streamingPreset": "NoStreaming",
            "retentionPeriod": "7"
        })

        try:
            print(f"  ⏳ שולח בקשה ל-Video Indexer...")
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            video_id = data.get("id") or data.get("videoId")

            if not video_id:
                raise RuntimeError(f"העלאה נכשלה: {data}")

            print(f"  ✅ הועלה בהצלחה, מזהה וידאו: {video_id}")
            return video_id

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"שגיאה בהעלאת הוידאו: {str(e)}")

    def wait_for_indexing(self, video_id: str, interval: int = 10, max_wait_minutes: int = 180) -> Dict:
        """המתנה לסיום עיבוד הוידאו ב-Video Indexer"""
        print(f"⏳ ממתין לסיום עיבוד הוידאו {video_id}...")

        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while True:
            try:
                params = self._get_params_with_token()
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                state = data.get("state")
                print(f"  📊 מצב עיבוד: {state}")

                if state == "Processed":
                    print("  ✅ עיבוד הושלם!")
                    return data
                elif state == "Failed":
                    raise RuntimeError("עיבוד הוידאו נכשל")

                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_seconds:
                    raise TimeoutError(f"עיבוד הוידאו לקח יותר מ-{max_wait_minutes} דקות")

                time.sleep(interval)

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"שגיאה בבדיקת מצב העיבוד: {str(e)}")

    def extract_transcript_with_timestamps(self, index_json: Dict) -> List[Dict]:
        """חילוץ טרנסקריפט עם חותמות זמן"""
        transcript_items = (
            index_json
            .get("videos", [{}])[0]
            .get("insights", {})
            .get("transcript", [])
        )

        transcript_segments = []

        for item in transcript_items:
            text = item.get("text", "")
            instances = item.get("instances", [])

            if text and instances:
                first_instance = instances[0]
                start_time = first_instance.get("start", "00:00:00")
                end_time = first_instance.get("end", "00:00:00")

                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)

                segment = {
                    "text": text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration": end_seconds - start_seconds,
                    "confidence": item.get("confidence", 0.9)
                }
                transcript_segments.append(segment)

        return transcript_segments

    def merge_segments_by_duration(self, segments: List[Dict], max_duration_seconds: int = 30) -> List[Dict]:
        """איחוד סגמנטים לסגמנטים ארוכים יותר"""
        if not segments:
            return []

        merged_segments = []
        current_segment = None

        for segment in segments:
            if current_segment is None:
                current_segment = {
                    "text": segment["text"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "start_seconds": segment["start_seconds"],
                    "end_seconds": segment["end_seconds"],
                    "duration": segment["duration"],
                    "confidence": segment["confidence"]
                }
            else:
                potential_duration = segment["end_seconds"] - current_segment["start_seconds"]

                if potential_duration <= max_duration_seconds:
                    current_segment["text"] += " " + segment["text"]
                    current_segment["end_time"] = segment["end_time"]
                    current_segment["end_seconds"] = segment["end_seconds"]
                    current_segment["duration"] = current_segment["end_seconds"] - current_segment["start_seconds"]
                    current_segment["confidence"] = (current_segment["confidence"] + segment["confidence"]) / 2
                else:
                    merged_segments.append(current_segment)
                    current_segment = {
                        "text": segment["text"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "start_seconds": segment["start_seconds"],
                        "end_seconds": segment["end_seconds"],
                        "duration": segment["duration"],
                        "confidence": segment["confidence"]
                    }

        if current_segment is not None:
            merged_segments.append(current_segment)

        print(f"  🔗 איחוד סגמנטים: {len(segments)} → {len(merged_segments)} (מקס {max_duration_seconds} שניות)")
        return merged_segments

    def create_textual_summary(self, video_id: str, deployment_name: str = "gpt-4o") -> str:
        """יצירת סיכום טקסטואלי באמצעות GPT"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual"
        params = self._get_params_with_token({
            "deploymentName": deployment_name,
            "length": "Long",
            "style": "Formal",
            "includedFrames": "None",
            "addToEndOfSummaryInstructions": "כתוב סיכום מפורט של השיעור בעברית. הצג את החומר בסדר הכרונולוגי שבו נלמד. לכל נושא מרכזי, תן הגדרות ברורות למונחים חדשים והסבר את הנקודה המרכזית במשפט אחד. כתוב בטון פדגוגי כך שסטודנט יוכל לשלוט בחומר מהסיכום לבדו. סיים ברשימת פעולות קונקרטיות או המלצות לימוד לסטודנטים. בסוף הוסף שורה אחת עם מילה אחת בלבד: 'מתמטי' או 'הומני' כדי לסווג אם זה קורס מתמטי/טכני או הומני."
        })

        try:
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()
            summary_id = resp.json().get("id")
            if not summary_id:
                raise RuntimeError(f"יצירת סיכום נכשלה: {resp.text}")
            print(f"  ✅ נוצר סיכום עם מזהה: {summary_id}")
            return summary_id
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                print(f"  ⚠️ יצירת סיכום GPT נכשלה (400 Bad Request)")
                print(f"  ⚠️ יצירת סיכום GPT נכשלה (400 Bad Request)")
                print(f"  📝 תגובה: {e.response.text}")
                raise RuntimeError(f"סיכום GPT לא זמין: {e.response.text}")
            else:
                raise

    def get_textual_summary(self, video_id: str, summary_id: str) -> str:
        """קבלת הסיכום הטקסטואלי לאחר שהוא מוכן"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual/{summary_id}"

        while True:
            params = self._get_params_with_token()
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state")
            if state == "Processed":
                summary = data.get("summary", "")
                print(f"  ✅ הסיכום מוכן. אורך: {len(summary)} תווים")
                return summary
            elif state == "Failed":
                raise RuntimeError(f"יצירת הסיכום נכשלה: {data}")
            else:
                print(f"  ⏳ מצב הסיכום: {state} — ממתין...")
                time.sleep(10)

    def delete_video(self, video_id: str) -> bool:
        """מחיקת וידאו מ-Video Indexer כדי לנקות קונטיינרים מיותרים"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}"

        try:
            params = self._get_params_with_token()
            resp = requests.delete(url, params=params, timeout=30)
            resp.raise_for_status()

            print(f"  🗑️ הוידאו נמחק מ-Video Indexer: {video_id}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ שגיאה במחיקת הוידאו מ-Video Indexer: {str(e)}")
            return False

    def extract_video_metadata(self, index_json: Dict) -> Dict:
        """חילוץ מטא-דאטה מהוידאו"""
        vid_info = index_json.get('videos', [{}])[0]
        insights = vid_info.get('insights', {})

        # משך זמן
        duration_sec = vid_info.get('durationInSeconds', 0)
        if not duration_sec:
            duration_obj = insights.get('duration')
            if isinstance(duration_obj, dict):
                duration_sec = duration_obj.get('time', 0)
            elif isinstance(duration_obj, str):
                duration_sec = self._time_to_seconds(duration_obj)
            else:
                duration_sec = insights.get('durationInSeconds', 0)

        # מילות מפתח ונושאים
        keywords = [kw.get('text') for kw in insights.get('keywords', []) if kw.get('text')]
        topics = [tp.get('name') for tp in insights.get('topics', []) if tp.get('name')]

        # OCR
        ocr_texts = []
        if 'ocr' in insights:
            ocr_texts = [o.get('text') for o in insights.get('ocr', []) if o.get('text')]

        # דוברים
        speakers = []
        if 'speakers' in insights:
            speakers = [s.get('name') for s in insights.get('speakers', []) if s.get('name')]
        if not speakers and 'speakers' in insights:
            speakers = [f"דובר #{s.get('id', i + 1)}" for i, s in enumerate(insights.get('speakers', []))]

        metadata = {
            'video_id': index_json.get('id', ''),
            'name': vid_info.get('name', ''),
            'description': index_json.get('description', ''),
            'duration': self._seconds_to_hhmmss(int(duration_sec)) if duration_sec else 'לא זמין',
            'language': insights.get('sourceLanguage', 'he-IL'),
            'keywords': keywords,
            'topics': topics,
            'ocr': ocr_texts,
            'speakers': speakers if speakers else ['מרצה'],
            'created_date': datetime.now().isoformat()
        }

        print(f"  📊 חולץ מטא-דאטה:")
        print(f"    - משך זמן: {metadata['duration']}")
        print(f"    - מילות מפתח: {len(keywords)} נמצאו")
        print(f"    - נושאים: {len(topics)} נמצאו")
        print(f"    - טקסטי OCR: {len(ocr_texts)} נמצאו")

        return metadata

    def _time_to_seconds(self, time_str: str) -> int:
        """המרת זמן לשניות"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    return int(hours * 3600 + minutes * 60 + seconds)
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    return int(minutes * 60 + seconds)
            return int(float(time_str))
        except:
            return 0

    def _seconds_to_hhmmss(self, seconds: int) -> str:
        """המרת שניות לפורמט HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def parse_insights_to_md(self, structured_data: Dict) -> str:
        """המרת נתוני הוידאו לפורמט Markdown"""
        md_content = []

        # כותרת ראשית
        md_content.append(f"# {structured_data.get('name', 'דוח ניתוח וידאו')}")
        md_content.append("")

        # מטא-דאטה
        md_content.append("## 📊 פרטי הוידאו")
        md_content.append(f"- **מזהה וידאו**: {structured_data.get('id', 'לא זמין')}")
        md_content.append(f"- **משך זמן**: {structured_data.get('duration', 'לא זמין')}")
        md_content.append(f"- **שפה**: {structured_data.get('language', 'לא זמין')}")
        md_content.append(f"- **דוברים**: {', '.join(structured_data.get('speakers', []))}")
        md_content.append(f"- **תאריך יצירה**: {structured_data.get('created_date', 'לא זמין')}")
        md_content.append("")

        # מילות מפתח
        keywords = structured_data.get('keywords', [])
        if keywords:
            md_content.append("## 🔍 מילות מפתח")
            md_content.append(", ".join(f"`{kw}`" for kw in keywords))
            md_content.append("")

        # נושאים
        topics = structured_data.get('topics', [])
        if topics:
            md_content.append("## 🏷️ נושאים")
            md_content.append(", ".join(f"`{topic}`" for topic in topics))
            md_content.append("")

        # OCR
        ocr_texts = structured_data.get('ocr', [])
        if ocr_texts:
            md_content.append("## 👁️ טקסט שחולץ מהוידאו (OCR)")
            for i, ocr_text in enumerate(ocr_texts, 1):
                md_content.append(f"{i}. {ocr_text}")
            md_content.append("")

        # סיכום השיעור
        summary_text = structured_data.get('summary_text', '')
        if summary_text:
            # חילוץ סוג המקצוע מהסיכום
            subject_type = "לא זוהה"
            summary_lines = summary_text.strip().split('\n')
            last_line = summary_lines[-1].strip() if summary_lines else ""

            if last_line in ['מתמטי', 'הומני']:
                subject_type = last_line
                # הסרת השורה האחרונה מהסיכום
                summary_text = '\n'.join(summary_lines[:-1]).strip()

            # הוספת סוג המקצוע
            md_content.append(f"## 🎓 סוג מקצוע")
            md_content.append(subject_type)
            md_content.append("")

            # הוספת הסיכום
            md_content.append("## 📝 סיכום השיעור")
            md_content.append(summary_text)
            md_content.append("")

        # תיאור
        if structured_data.get('description'):
            md_content.append("## 📝 תיאור")
            md_content.append(structured_data['description'])
            md_content.append("")

        # טרנסקריפט מלא
        md_content.append("## 📄 טרנסקריפט מלא")
        md_content.append(structured_data.get('full_transcript', 'טרנסקריפט לא זמין'))
        md_content.append("")

        # טרנסקריפט עם חותמות זמן
        transcript_segments = structured_data.get('transcript_segments', [])
        if transcript_segments:
            md_content.append("## ⏰ טרנסקריפט עם חותמות זמן")
            md_content.append("")

            for segment in transcript_segments:
                start_time = segment.get('start_time', '00:00:00')
                text = segment.get('text', '')
                md_content.append(f"**[{start_time}]** {text}")
                md_content.append("")

        return "\n".join(md_content)


    def process_video_to_md(self, course_id: str, section_id: str, file_id: int, video_name: str, video_url: str, merge_segments_duration: Optional[int] = 30) -> str | None:
        """
        עיבוד וידאו מ-blob storage ליצירת קובץ markdown

        Args:
            course_id: מזהה הקורס
            section_id: מזהה הסקציה
            file_id: מזהה הקובץ
            video_name: שם הוידאו (ייכנס לתמלול)
            video_url: נתיב הוידאו ב-blob storage
            merge_segments_duration: משך זמן מקסימלי בשניות לאיחוד סגמנטים

        Returns:
            נתיב הקובץ ב-blob storage או None אם נכשל
        """
        # יצירת מנהלי blob - אחד לקריאה מ-raw-data ואחד לכתיבה ל-processeddata
        blob_manager_read = BlobManager(container_name="raw-data")
        blob_manager_write = BlobManager(container_name="processeddata")

        # בדיקת סיומת הקובץ
        file_ext = os.path.splitext(video_url)[1].lower()
        if file_ext not in self.supported_formats:
            print(f"❌ פורמט וידאו לא נתמך: {video_url}")
            return None

        # יצירת SAS URL לוידאו מקונטיינר raw-data
        print(f"🔗 יוצר SAS URL לוידאו מקונטיינר raw-data: {video_url}")
        video_sas_url = blob_manager_read.generate_sas_url(video_url, hours=4)

        if not video_sas_url:
            print(f"❌ נכשלה יצירת SAS URL לוידאו: {video_url}")
            return None

        print(f"🔄 מעבד וידאו: {video_name}")

        try:
            print(f"\n🎬 מתחיל עיבוד וידאו ל-MD: {video_name}")

            # העלאה ועיבוד ל-Video Indexer עם שם הוידאו
            video_id = self.upload_video_from_url(video_sas_url, video_name)
            index_data = self.wait_for_indexing(video_id)

            # יצירת סיכום GPT
            print("  📝 יוצר סיכום GPT...")
            summary_text = ""
            try:
                summary_id = self.create_textual_summary(video_id)
                summary_text = self.get_textual_summary(video_id, summary_id)
                print(f"  ✅ התקבל סיכום באורך: {len(summary_text)} תווים")
            except Exception as e:
                print(f"  ⚠️ יצירת סיכום GPT נכשלה, ממשיך בלי סיכום: {e}")

            # חילוץ טרנסקריפט
            transcript_segments = self.extract_transcript_with_timestamps(index_data)

            # איחוד סגמנטים אם נדרש
            if merge_segments_duration:
                print(f"  🔗 מאחד סגמנטים למקסימום {merge_segments_duration} שניות...")
                transcript_segments = self.merge_segments_by_duration(transcript_segments, merge_segments_duration)

            # חילוץ מטא-דאטה
            metadata = self.extract_video_metadata(index_data)

            # יצירת מבנה נתונים מובנה עם שם הוידאו
            structured_data = {
                "id": str(file_id),  # שימוש ב-file_id כמזהה במקום video_id
                "name": video_name,  # שימוש בשם שהועבר
                **metadata,
                "transcript_segments": transcript_segments,
                "full_transcript": " ".join([seg["text"] for seg in transcript_segments]),
                "segment_start_times": [seg["start_time"] for seg in transcript_segments],
                "segment_start_seconds": [seg["start_seconds"] for seg in transcript_segments],
                "summary_text": summary_text
            }

            # המרה ל-markdown
            md_content = self.parse_insights_to_md(structured_data)

            print(f"  ✅ עיבוד הושלם בהצלחה!")
            print(f"  📊 נמצאו {len(transcript_segments)} קטעי טרנסקריפט")

        except Exception as e:
            print(f"❌ נכשל עיבוד הוידאו: {str(e)}")
            return None

        # יצירת נתיב היעד לפי המבנה: CourseID/SectionID/Videos_md/FileID.md
        target_blob_path = f"{course_id}/{section_id}/Videos_md/{file_id}.md"

        print(f"📤 מעלה לקונטיינר processeddata: {target_blob_path}")

        # שמירה לקונטיינר processeddata
        success = blob_manager_write.upload_text_to_blob(
            text_content=md_content,
            blob_name=target_blob_path
        )

        if success:
            print(f"✅ הקובץ הועלה בהצלחה לקונטיינר processeddata: {target_blob_path}")

            # ניקוי: מחיקת הוידאו מ-Video Indexer כדי לנקות קונטיינרים מיותרים
            print("🧹 מנקה קונטיינרים מיותרים...")
            self.delete_video(video_id)

            return target_blob_path
        else:
            print(f"❌ נכשלה העלאת הקובץ לקונטיינר processeddata")
            return None


if __name__ == "__main__":
    # עיבוד וידאו מ-blob storage עם פרמטרים חדשים
    course_id = "CS101"
    section_id = "Section1"
    file_id = 2
    video_name = "שיעור ראשון - חתוך"
    video_url = "L1_091004f349688522f773afc884451c9af6da18fb_Trim.mp4"

    print(f"🧪 מעבד וידאו: {video_name}")
    print(f"📍 CourseID: {course_id}, SectionID: {section_id}, FileID: {file_id}")
    print(f"🔗 VideoURL: {video_url}")

    try:
        manager = VideoIndexerManager()
        result = manager.process_video_to_md(course_id, section_id, file_id, video_name, video_url, merge_segments_duration=10)

        if result:
            print(f"\n🎉 הוידאו עובד בהצלחה: {result}")
            print(f"📁 הקובץ נשמר במבנה: {course_id}/{section_id}/Videos_md/{file_id}.md")
        else:
            print(f"\n❌ נכשל עיבוד הוידאו: {video_name}")

    except Exception as e:
        print(f"❌ שגיאה בעיבוד הוידאו: {e}")
