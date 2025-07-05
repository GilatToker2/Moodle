#################
######## Wroking !!!!!!!!
###############

import requests
import time
import os
import json
from datetime import datetime
from typing import Optional, Dict, List
from config import VIDEO_INDEXER_TOKEN, VIDEO_INDEXER_ACCOUNT_ID, VIDEO_INDEXER_LOCATION


class VideoIndexerManager:
    """
    Manages video upload and processing through Azure Video Indexer

    This class provides methods to upload videos, extract transcripts,
    keywords, topics, OCR text, and generate comprehensive markdown reports.
    """
    def __init__(self):
        self.token = VIDEO_INDEXER_TOKEN
        self.account_id = VIDEO_INDEXER_ACCOUNT_ID
        self.location = VIDEO_INDEXER_LOCATION
        self.supported_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']

    def upload_video(self, video_path: str, video_name: Optional[str] = None) -> str:
        """
        Upload a video file to Azure Video Indexer

        Args:
            video_path: Path to the local video file
            video_name: Optional custom name for the video (timestamp will be added)
        """

        print(f"📤 מעלה וידאו: {os.path.basename(video_path)}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"קובץ הוידאו לא נמצא: {video_path}")

        # בדיקת פורמט הקובץ
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"פורמט קובץ לא נתמך: {file_ext}")

        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos"

        # יצירת שם לוידאו עם timestamp כדי למנוע התנגשויות
        if not video_name:
            video_filename = os.path.basename(video_path)
            video_name_base = os.path.splitext(video_filename)[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # פשוט יותר
            video_name = f"{video_name_base}_{timestamp}"
        else:
            # גם אם נתן שם, נוסיף timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_name = f"{video_name}_{timestamp}"

        # נסיונות חוזרים להעלאה
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(video_path, 'rb') as f:
                    files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
                    params = {
                        "name": video_name,
                        "privacy": "Private",
                        "language": "he-IL",
                        "accessToken": self.token
                    }

                    print(f"  ⏳ מעלה קובץ... (נסיון {attempt + 1}/{max_retries})")
                    resp = requests.post(url, files=files, params=params, timeout=300)  # 5 דקות timeout
                    resp.raise_for_status()

                    data = resp.json()
                    video_id = data.get("id") or data.get("videoId")

                    if not video_id:
                        raise RuntimeError(f"העלאה נכשלה עבור {video_path}, תגובה לא צפויה:\n{data}")

                    print(f"  ✅ הועלה בהצלחה, מזהה וידאו: {video_id}")
                    return video_id

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # המתנה הולכת וגדלה
                    print(f"  ⚠️ נסיון {attempt + 1} נכשל: {str(e)}")
                    print(f"  ⏳ ממתין {wait_time} שניות לפני נסיון חוזר...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"שגיאה בהעלאת הוידאו אחרי {max_retries} נסיונות: {str(e)}")

    def wait_for_indexing(self, video_id: str, interval: int = 10, max_wait_minutes: int = 180) -> Dict:

        """Wait for video processing to complete in Azure Video Indexer"""
        print(f"⏳ ממתין לסיום עיבוד הוידאו {video_id}...")

        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        params = {"accessToken": self.token}

        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while True:
            try:
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

                # בדיקת timeout
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
                # קבלת המופע הראשון לזמנים
                first_instance = instances[0]
                start_time = first_instance.get("start", "00:00:00")
                end_time = first_instance.get("end", "00:00:00")

                # המרת זמן לשניות
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

    def extract_video_metadata(self, index_json: Dict) -> Dict:
        """חילוץ מטא-דאטה מלא מהוידאו"""
        vid_info = index_json.get('videos', [{}])[0]
        insights = vid_info.get('insights', {})

        # Debug: print the structure to understand what we're getting
        print(f"  🔍 Debug - Available insights keys: {list(insights.keys())}")
        print(f"  🔍 Debug - Top level keys: {list(index_json.keys())}")

        # Debug specific fields we're looking for
        if 'summarizedInsights' in index_json:
            print(f"  🔍 Debug - summarizedInsights keys: {list(index_json['summarizedInsights'].keys())}")
        else:
            print("  ⚠️ Debug - No summarizedInsights found")

        if 'ocr' in insights:
            print(f"  🔍 Debug - OCR found with {len(insights['ocr'])} items")
        else:
            print("  ⚠️ Debug - No OCR found in insights")

        # duration - try multiple possible locations
        duration_sec = vid_info.get('durationInSeconds', 0)
        if not duration_sec:
            # Try to get from insights.duration
            duration_obj = insights.get('duration')
            if isinstance(duration_obj, dict):
                duration_sec = duration_obj.get('time', 0)
            elif isinstance(duration_obj, str):
                # Sometimes duration is a string like "0:00:16.12"
                duration_sec = self._time_to_seconds(duration_obj)
            else:
                duration_sec = insights.get('durationInSeconds', 0)

        # keywords & topics
        keywords = [kw.get('text') for kw in insights.get('keywords', []) if kw.get('text')]
        topics = [tp.get('name') for tp in insights.get('topics', []) if tp.get('name')]

        # OCR - try different possible field names
        ocr_texts = []
        if 'ocr' in insights:
            ocr_texts = [o.get('text') for o in insights.get('ocr', []) if o.get('text')]
        elif 'visualContentModeration' in insights:
            # Sometimes OCR is under visual content moderation
            visual_content = insights.get('visualContentModeration', {})
            if 'instances' in visual_content:
                for instance in visual_content['instances']:
                    if 'text' in instance:
                        ocr_texts.append(instance['text'])

        # summary text - try multiple locations
        summary = (index_json.get('summarizedInsights', {}).get('summary') or
                   insights.get('summary') or
                   insights.get('textualContentModeration', {}).get('summary', ''))

        # speakers
        speakers = []
        if 'speakers' in insights:
            speakers = [s.get('name') for s in insights.get('speakers', []) if s.get('name')]

        # If no named speakers, try to get speaker IDs
        if not speakers and 'speakers' in insights:
            speakers = [f"Speaker #{s.get('id', i + 1)}" for i, s in enumerate(insights.get('speakers', []))]

        metadata = {
            'video_id': index_json.get('id', ''),
            'name': vid_info.get('name', ''),
            'description': index_json.get('description', ''),
            'duration': self._seconds_to_hhmmss(int(duration_sec)) if duration_sec else 'N/A',
            'language': insights.get('sourceLanguage', 'he-IL'),
            'keywords': keywords,
            'topics': topics,
            'ocr': ocr_texts,
            'summary': summary,
            'speakers': speakers if speakers else ['מרצה'],
            'created_date': datetime.now().isoformat(),
            'video_indexer_url': f"https://www.videoindexer.ai/embed/player/{self.account_id}/{index_json.get('id', '')}?location={self.location}"
        }

        # Debug output
        print(f"  📊 Extracted metadata:")
        print(f"    - Duration: {metadata['duration']}")
        print(f"    - Keywords: {len(keywords)} found")
        print(f"    - Topics: {len(topics)} found")
        print(f"    - OCR texts: {len(ocr_texts)} found")
        print(f"    - Summary length: {len(summary)} chars")

        return metadata

    def create_textual_summary(self, video_id: str, deployment_name: str, length="Long", style="Formal",
                               included_frames="None") -> str:
        """
        Create a textual summary job using Azure Video Indexer connected to Azure OpenAI.
        Returns the summaryId for polling.
        """
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual"
        params = {
            "deploymentName": deployment_name,
            "length": length,  # short | medium | long
            "style": style,  # neutral | casual | formal
            "includedFrames": included_frames,  # None | Keyframes
            "accessToken": self.token,
            "addToEndOfSummaryInstructions": "Compose an in-depth, chronological summary of the lecture in 1 000–1 500 words. \
            Present the material in the exact order it was taught. For every major topic, \
            (1) give clear definitions of new terms or concepts, and (2) state the key takeaway \
            in one sentence. Write in a pedagogical tone so a student can master the lesson \
            from this summary alone. Conclude with a bulleted list of concrete action items \
            or study recommendations for the students."
        }

        try:
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()
            summary_id = resp.json().get("id")
            if not summary_id:
                raise RuntimeError(f"Create Summary failed: {resp.text}")
            print(f"  ✅ Summary job created with ID: {summary_id}")
            return summary_id
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                print(f"  ⚠️ GPT Summary creation failed (400 Bad Request). This might be due to:")
                print(f"    - Incorrect deployment name: '{deployment_name}'")
                print(f"    - GPT summarization not enabled for this account")
                print(f"    - Invalid parameters")
                print(f"  📝 Response: {e.response.text}")
                raise RuntimeError(f"GPT Summary not available: {e.response.text}")
            else:
                raise

    def get_textual_summary(self, video_id: str, summary_id: str) -> str:
        """
        Poll the summary job until it's ready, then return the generated summary text.
        """
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual/{summary_id}"
        params = {
            "accessToken": self.token
        }
        while True:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state")
            if state == "Processed":
                summary = data.get("summary", "")
                print(f"  ✅ Summary is ready. Length: {len(summary)} chars")
                return summary
            elif state == "Failed":
                raise RuntimeError(f"Summary generation failed: {data}")
            else:
                print(f"  ⏳ Summary state: {state} — waiting...")
                time.sleep(10)

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

    def _save_summary_to_file(self, video_id: str, summary_text: str, video_name: str) -> str:
        """שמירת הסיכום כקובץ MD נפרד בתיקיית video_summary"""
        # יצירת תיקיית video_summary אם לא קיימת
        summary_dir = "video_summary"
        os.makedirs(summary_dir, exist_ok=True)

        # יצירת שם קובץ בטוח
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"{safe_video_name}_{video_id}_{timestamp}_summary.md"
        summary_path = os.path.join(summary_dir, summary_filename)

        # יצירת תוכן הקובץ
        summary_content = f"# סיכום וידאו: {video_name}\n\n"
        summary_content += f"**Video ID**: {video_id}\n"
        summary_content += f"**תאריך יצירה**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        summary_content += "## סיכום\n\n"
        summary_content += summary_text

        # שמירת הקובץ
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_content)

        print(f"  ✅ הסיכום נשמר בקובץ: {summary_path}")
        return summary_path

    def process_video_complete(self, video_path: str, video_name: Optional[str] = None) -> Dict:
        """עיבוד מלא של וידאו - העלאה, המתנה לעיבוד וחילוץ נתונים"""
        print(f"\n🎬 מתחיל עיבוד מלא של: {os.path.basename(video_path)}")

        try:
            # שלב 1: העלאת הוידאו
            video_id = self.upload_video(video_path, video_name)

            # שלב 2: המתנה לסיום העיבוד
            index_data = self.wait_for_indexing(video_id)

            # ⭐️ שלב 2.5: הפעלת הסיכום מבוסס GPT
            print("  📝 Generating GPT-based summary...")
            deployment_name = "gpt-4o"  # replace with your actual deployment name
            summary_id = self.create_textual_summary(video_id, deployment_name=deployment_name)
            summary_text = self.get_textual_summary(video_id, summary_id)
            print(f"  ✅ Received summary with length: {len(summary_text)}")

            # שלב 3: חילוץ טרנסקריפט
            print("  📝 מחלץ טרנסקריפט...")
            transcript_segments = self.extract_transcript_with_timestamps(index_data)

            # שלב 4: חילוץ מטא-דאטה
            print("  📊 מחלץ מטא-דאטה...")
            metadata = self.extract_video_metadata(index_data)

            # 🟢 שמירת הסיכום כקובץ MD נפרד
            self._save_summary_to_file(video_id, summary_text, metadata.get('name', 'Unknown Video'))

            # שלב 5: יצירת מבנה נתונים מובנה (ללא הסיכום במטאדאטה)
            structured_data = {
                "id": video_id,
                **metadata,
                "transcript_segments": transcript_segments,
                "full_transcript": " ".join([seg["text"] for seg in transcript_segments]),
                "segment_start_times": [seg["start_time"] for seg in transcript_segments],
                "segment_start_seconds": [seg["start_seconds"] for seg in transcript_segments]
            }

            print(f"  ✅ עיבוד הושלם בהצלחה!")
            print(f"  📊 נמצאו {len(transcript_segments)} קטעי טרנסקריפט")

            return structured_data

        except Exception as e:
            print(f"  ❌ שגיאה בעיבוד הוידאו: {str(e)}")
            raise

    def process_video_from_path(self, video_path: str) -> str:
        """פונקציה נוחה לעיבוד וידאו מנתיב קובץ"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"קובץ הוידאו לא נמצא: {video_path}")

        # יצירת שם וידאו מהקובץ
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]

        # עיבוד הוידאו
        structured_data = self.process_video_complete(video_path, video_name)

        return structured_data["video_id"]

    def parse_insights_to_md(self, structured_data: Dict) -> str:
        """המרת נתוני הוידאו לפורמט Markdown מתקדם"""
        md_content = []

        # כותרת ראשית
        md_content.append(f"# {structured_data.get('name', 'Video Analysis Report')}")
        md_content.append("")

        # מטא-דאטה מורחבת
        md_content.append("## 📊 Video Information")
        md_content.append(f"- **Video ID**: {structured_data.get('video_id', 'N/A')}")
        md_content.append(f"- **Duration**: {structured_data.get('duration', 'N/A')}")
        md_content.append(f"- **Language**: {structured_data.get('language', 'N/A')}")
        md_content.append(f"- **Speakers**: {', '.join(structured_data.get('speakers', []))}")
        md_content.append(f"- **Created**: {structured_data.get('created_date', 'N/A')}")

        if structured_data.get('video_indexer_url'):
            md_content.append(f"- **Video Indexer URL**: [View Video]({structured_data['video_indexer_url']})")

        md_content.append("")

        # מילות מפתח
        keywords = structured_data.get('keywords', [])
        if keywords:
            md_content.append("## 🔍 Keywords")
            md_content.append(", ".join(f"`{kw}`" for kw in keywords))
            md_content.append("")

        # נושאים
        topics = structured_data.get('topics', [])
        if topics:
            md_content.append("## 🏷️ Topics")
            md_content.append(", ".join(f"`{topic}`" for topic in topics))
            md_content.append("")

        # OCR - טקסט שחולץ מהוידאו
        ocr_texts = structured_data.get('ocr', [])
        if ocr_texts:
            md_content.append("## 👁️ OCR - Text Extracted from Video")
            for i, ocr_text in enumerate(ocr_texts, 1):
                md_content.append(f"{i}. {ocr_text}")
            md_content.append("")

        # תיאור אם קיים
        if structured_data.get('description'):
            md_content.append("## 📝 Description")
            md_content.append(structured_data['description'])
            md_content.append("")

        # טרנסקריפט מלא
        md_content.append("## 📄 Full Transcript")
        md_content.append(structured_data.get('full_transcript', 'No transcript available'))
        md_content.append("")

        # טרנסקריפט עם חותמות זמן
        transcript_segments = structured_data.get('transcript_segments', [])
        if transcript_segments:
            md_content.append("## ⏰ Transcript with Timestamps")
            md_content.append("")

            for segment in transcript_segments:
                start_time = segment.get('start_time', '00:00:00')
                text = segment.get('text', '')
                md_content.append(f"**[{start_time}]** {text}")
                md_content.append("")

        return "\n".join(md_content)

    def get_supported_formats(self) -> List[str]:
        """קבלת רשימת פורמטים נתמכים"""
        return self.supported_formats.copy()


def process_video_to_md(file_path: str) -> str:
    """
    Main entry point function that processes a video file and returns markdown content.

    Args:
        file_path: Path to the local video file

    Returns:
        Markdown string containing the video transcript and metadata
    """
    manager = VideoIndexerManager()

    # Upload video and wait for indexing
    video_id = manager.upload_video(file_path)
    index_data = manager.wait_for_indexing(video_id)

    # ⭐️ Generate GPT-based summary
    print("  📝 Generating GPT-based summary...")
    deployment_name = "gpt-4o"  # replace with your actual deployment name
    summary_id = manager.create_textual_summary(video_id, deployment_name=deployment_name)
    summary_text = manager.get_textual_summary(video_id, summary_id)
    print(f"  ✅ Received summary with length: {len(summary_text)}")

    # Extract transcript and metadata
    transcript_segments = manager.extract_transcript_with_timestamps(index_data)
    metadata = manager.extract_video_metadata(index_data)

    # 🟢 שמירת הסיכום כקובץ MD נפרד
    manager._save_summary_to_file(video_id, summary_text, metadata.get('name', 'Unknown Video'))

    # Create structured data (ללא הסיכום במטאדאטה)
    structured_data = {
        "id": video_id,
        **metadata,
        "transcript_segments": transcript_segments,
        "full_transcript": " ".join([seg["text"] for seg in transcript_segments]),
        "segment_start_times": [seg["start_time"] for seg in transcript_segments],
        "segment_start_seconds": [seg["start_seconds"] for seg in transcript_segments]
    }

    # Convert to markdown
    md_string = manager.parse_insights_to_md(structured_data)

    return md_string

def test_video_processing(video_path: str):
    """בדיקת עיבוד וידאו פשוטה"""
    print(f"🎬 מעבד וידאו: {video_path}")

    manager = VideoIndexerManager()
    video_id = manager.process_video_from_path(video_path)

    print(f"✅ הושלם! מזהה: {video_id}")
    return video_id


if __name__ == "__main__":
    # Example usage with local video file
    video_file_path = r"Lectures\L9_18f0d24bb7e45223abf842cdc1274de65fc7d620 - Trim.mp4"

    try:
        md = process_video_to_md(video_file_path)

        # Create Videos_MD directory if it doesn't exist
        os.makedirs("Videos_MD", exist_ok=True)

        # Generate output filename based on video filename
        video_filename = os.path.basename(video_file_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]
        output_filename = f"{video_name_without_ext}.md"
        output_path = os.path.join("Videos_MD", output_filename)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Video processed successfully! Output saved to {output_path}")
    except Exception as e:
        print(f"❌ Error processing video: {e}")
