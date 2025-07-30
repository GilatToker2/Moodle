import requests
import time
import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from Config.config import (
    VIDEO_INDEXER_ACCOUNT_ID,
    VIDEO_INDEXER_LOCATION,
    VIDEO_INDEXER_SUB_ID,
    VIDEO_INDEXER_RG,
    VIDEO_INDEXER_VI_ACC,
)
from dotenv import load_dotenv
from VideoIndexerClient.VideoIndexerClient import VideoIndexerClient
from VideoIndexerClient.Consts import Consts
from Source.Services.blob_manager import BlobManager
from Config.logging_config import setup_logging

logger = setup_logging()


class VideoIndexerManager:
    """
    Video processing manager using Azure Video Indexer
    Specializes in processing videos from blob storage and creating markdown files
    """

    def __init__(self):
        self.account_id = VIDEO_INDEXER_ACCOUNT_ID
        self.location = VIDEO_INDEXER_LOCATION
        self.subscription_id = VIDEO_INDEXER_SUB_ID
        self.resource_group = VIDEO_INDEXER_RG
        self.account_name = VIDEO_INDEXER_VI_ACC
        self.supported_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']

        self._access_token = None
        self._token_expiry = None

        # Create VideoIndexer client for token refresh
        self._vi_client = None
        self._consts = None
        self._initialize_vi_client()

    def _initialize_vi_client(self):
        """Initialize VideoIndexer client for token refresh"""
        try:
            load_dotenv()

            self._consts = Consts(
                ApiVersion='2024-01-01',
                ApiEndpoint='https://api.videoindexer.ai',
                AzureResourceManager='https://management.azure.com',
                AccountName=self.account_name,
                ResourceGroup=self.resource_group,
                SubscriptionId=self.subscription_id
            )

            self._vi_client = VideoIndexerClient()
            logger.info("✅ VideoIndexer client initialized successfully")

        except Exception as e:
            logger.info(f"⚠️ Error initializing VideoIndexer client: {e}")
            self._vi_client = None

    def get_valid_token(self):
        """Get valid token - automatically refreshes if needed"""
        if self._should_refresh_token():
            self._refresh_token()

        return self._access_token

    def _should_refresh_token(self):
        """Check if token needs to be refreshed"""
        if not self._access_token:
            return True

        if not self._token_expiry:
            return True

        # Refresh 5 minutes before expiry
        refresh_time = self._token_expiry - timedelta(minutes=5)
        return datetime.utcnow() >= refresh_time

    def _refresh_token(self):
        """Refresh Video Indexer token"""
        if not self._vi_client or not self._consts:
            logger.info("⚠️ VideoIndexer client not available, using fixed token")
            return

        try:
            logger.info("🔄 Refreshing Video Indexer token...")

            # Get new tokens
            arm_token, vi_token, response = self._vi_client.authenticate_async(self._consts)

            if vi_token:
                self._access_token = vi_token

                # Extract token expiry time
                self._extract_token_expiry(vi_token)

                logger.info(f"✅ Token refreshed successfully. Length: {len(vi_token)}")
                if self._token_expiry:
                    current_time = datetime.utcnow()
                    logger.info(f"🕐 Current time: {current_time}")
                    logger.info(f"⏰ Expires at: {self._token_expiry}")
            else:
                logger.info("❌ No new token received")

        except Exception as e:
            logger.info(f"❌ Error refreshing token: {e}")

    def _extract_token_expiry(self, token):
        try:
            # Instead of decoding the token, simply set it as valid for one hour from now
            self._token_expiry = datetime.utcnow() + timedelta(hours=1)
            logger.info(f"📅 Token expiry time (estimated): {self._token_expiry}")

        except Exception as e:
            logger.info(f"⚠️ Error setting expiry time: {e}")

    def _get_params_with_token(self, additional_params=None):
        """Get parameters with access token."""
        token = self.get_valid_token()
        params = {"accessToken": token}
        if additional_params:
            params.update(additional_params)
        return params

    def upload_video_from_url(self, video_sas_url: str, video_name: str) -> str:
        """
        Upload video to Video Indexer using SAS URL

        Args:
            video_sas_url: SAS URL of the video in blob storage
            video_name: Video name in Video Indexer
        """
        logger.info(f"📤 Uploading video: {video_name}")

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
            logger.info(f"  ⏳ Sending request to Video Indexer...")
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            video_id = data.get("id") or data.get("videoId")

            if not video_id:
                raise RuntimeError(f"Upload failed: {data}")

            logger.info(f"  ✅ Uploaded successfully, video ID: {video_id}")
            return video_id

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error uploading video: {str(e)}")

    async def wait_for_indexing(self, video_id: str, interval: int = 10, max_wait_minutes: int = 300) -> Dict:
        """Wait for video processing completion in Video Indexer (async)"""
        logger.info(f"⏳ Waiting for video processing completion {video_id}...")

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
                logger.info(f"  📊 Processing state: {state}")

                if state == "Processed":
                    logger.info("  ✅ Processing completed!")
                    return data
                elif state == "Failed":
                    raise RuntimeError("Video processing failed")

                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_seconds:
                    raise TimeoutError(f"Video processing took more than {max_wait_minutes} minutes")

                await asyncio.sleep(interval)

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error checking processing status: {str(e)}")

    def extract_transcript_with_timestamps(self, index_json: Dict) -> List[Dict]:
        """Extract transcript with timestamps"""
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
        """Merge segments into longer segments"""
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

        logger.info(
            f"  🔗 Merging segments: {len(segments)} → {len(merged_segments)} (max {max_duration_seconds} seconds)")
        return merged_segments

    def create_textual_summary(self, video_id: str, deployment_name: str = "gpt-4o") -> str:
        """Create textual summary using GPT"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual"
        params = self._get_params_with_token({
            "deploymentName": deployment_name,
            "length": "Long",
            "style": "Formal",
            "includedFrames": "All",
            "addToEndOfSummaryInstructions": "כתוב סיכום מפורט של השיעור בעברית. הצג את החומר בסדר הכרונולוגי שבו נלמד. לכל נושא מרכזי, תן הגדרות ברורות למונחים חדשים והסבר את הנקודה המרכזית במשפט אחד. כתוב בטון פדגוגי כך שסטודנט יוכל לשלוט בחומר מהסיכום לבדו. סיים ברשימת פעולות קונקרטיות או המלצות לימוד לסטודנטים. בסוף הוסף שורה אחת עם מילה אחת בלבד: 'מתמטי' או 'הומני' כדי לסווג אם זה קורס מתמטי או הומני."
        })

        try:
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()
            summary_id = resp.json().get("id")
            if not summary_id:
                raise RuntimeError(f"יצירת סיכום נכשלה: {resp.text}")
            logger.info(f"  ✅ נוצר סיכום עם מזהה: {summary_id}")
            return summary_id
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.info(f"  ⚠️ יצירת סיכום GPT נכשלה (400 Bad Request)")
                logger.info(f"  ⚠️ יצירת סיכום GPT נכשלה (400 Bad Request)")
                logger.info(f"  📝 תגובה: {e.response.text}")
                raise RuntimeError(f"סיכום GPT לא זמין: {e.response.text}")
            else:
                raise

    async def get_textual_summary(self, video_id: str, summary_id: str) -> str:
        """Get textual summary after it's ready (async)"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Summaries/Textual/{summary_id}"

        while True:
            params = self._get_params_with_token()
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state")
            if state == "Processed":
                summary = data.get("summary", "")
                logger.info(f"  ✅ Summary ready. Length: {len(summary)} characters")
                return summary
            elif state == "Failed":
                raise RuntimeError(f"Summary creation failed: {data}")
            else:
                logger.info(f"  ⏳ Summary state: {state} — waiting...")
                await asyncio.sleep(10)

    def delete_video(self, video_id: str) -> bool:
        """Delete video from Video Indexer to clean up unnecessary containers"""
        url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}"

        try:
            params = self._get_params_with_token()
            resp = requests.delete(url, params=params, timeout=30)
            resp.raise_for_status()

            logger.info(f"  🗑️ Video deleted from Video Indexer: {video_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.info(f"  ⚠️ Error deleting video from Video Indexer: {str(e)}")
            return False

    def extract_video_metadata(self, index_json: Dict) -> Dict:
        """Extract metadata from video"""
        vid_info = index_json.get('videos', [{}])[0]
        insights = vid_info.get('insights', {})

        # Duration
        duration_sec = vid_info.get('durationInSeconds', 0)
        if not duration_sec:
            duration_obj = insights.get('duration')
            if isinstance(duration_obj, dict):
                duration_sec = duration_obj.get('time', 0)
            elif isinstance(duration_obj, str):
                duration_sec = self._time_to_seconds(duration_obj)
            else:
                duration_sec = insights.get('durationInSeconds', 0)

        # Keywords and topics
        keywords = [kw.get('text') for kw in insights.get('keywords', []) if kw.get('text')]
        topics = [tp.get('name') for tp in insights.get('topics', []) if tp.get('name')]

        # OCR
        ocr_texts = []
        if 'ocr' in insights:
            ocr_texts = [o.get('text') for o in insights.get('ocr', []) if o.get('text')]

        # Speakers
        speakers = []
        if 'speakers' in insights:
            speakers = [s.get('name') for s in insights.get('speakers', []) if s.get('name')]
        if not speakers and 'speakers' in insights:
            speakers = [f"Speaker #{s.get('id', i + 1)}" for i, s in enumerate(insights.get('speakers', []))]

        metadata = {
            'video_id': index_json.get('id', ''),
            'name': vid_info.get('name', ''),
            'description': index_json.get('description', ''),
            'duration': self._seconds_to_hhmmss(int(duration_sec)) if duration_sec else 'Not available',
            'language': insights.get('sourceLanguage', 'he-IL'),
            'keywords': keywords,
            'topics': topics,
            'ocr': ocr_texts,
            'speakers': speakers if speakers else ['Lecturer'],
            'created_date': datetime.now().isoformat()
        }

        logger.info(f"  📊 Extracted metadata:")
        logger.info(f"    - Duration: {metadata['duration']}")
        logger.info(f"    - Keywords: {len(keywords)} found")
        logger.info(f"    - Topics: {len(topics)} found")
        logger.info(f"    - OCR texts: {len(ocr_texts)} found")

        return metadata

    def _time_to_seconds(self, time_str: str) -> int:
        """Convert time to seconds"""
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
        """Convert seconds to HH:MM:SS format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def parse_insights_to_md(self, structured_data: Dict) -> str:
        """Convert video data to Markdown format"""
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

    async def process_video_to_md(self, course_id: str, section_id: str, file_id: int, video_name: str, video_url: str,
                                  merge_segments_duration: Optional[int] = 30) -> str | None:
        """
        NON-BLOCKING: Process video from blob storage to create markdown file
        Returns target path immediately after uploading video, processing continues in background

        Args:
            course_id: Course identifier
            section_id: Section identifier
            file_id: File identifier
            video_name: Video name (will be included in transcription)
            video_url: Video path in blob storage
            merge_segments_duration: Maximum duration in seconds for merging segments

        Returns:
            File path in blob storage where final result will be saved, or None if upload failed
        """
        # Create blob manager for reading from raw-data
        blob_manager_read = BlobManager(container_name="raw-data")

        # Check file extension
        file_ext = os.path.splitext(video_url)[1].lower()
        if file_ext not in self.supported_formats:
            logger.info(f"❌ Unsupported video format: {video_url}")
            return None

        # Create SAS URL for video from raw-data container
        logger.info(f"🔗 Creating SAS URL for video from raw-data container: {video_url}")
        video_sas_url = blob_manager_read.generate_sas_url(video_url, hours=4)

        if not video_sas_url:
            logger.info(f"❌ Failed to create SAS URL for video: {video_url}")
            return None

        logger.info(f"🔄 Starting NON-BLOCKING video processing: {video_name}")

        try:
            logger.info(f"📤 Uploading video to Video Indexer: {video_name}")

            # Upload video to Video Indexer (this is quick)
            video_id = self.upload_video_from_url(video_sas_url, video_name)

            # Create target path immediately
            target_blob_path = f"{course_id}/{section_id}/Videos_md/{file_id}.md"

            logger.info(f"✅ Video uploaded successfully! Video ID: {video_id}")
            logger.info(f"🚀 Starting background processing for: {video_name}")
            logger.info(f"📁 Final result will be saved to: {target_blob_path}")

            # Start background processing as async task
            asyncio.create_task(
                self._background_process_video(
                    video_id,
                    course_id,
                    section_id,
                    file_id,
                    video_name,
                    merge_segments_duration
                )
            )

            # Return target path immediately - processing continues in background
            return target_blob_path

        except Exception as e:
            logger.info(f"❌ Video upload failed: {str(e)}")
            return None

    async def _background_process_video(self, video_id: str, course_id: str, section_id: str, file_id: int,
                                        video_name: str, merge_segments_duration: Optional[int] = 30):
        """Background processing of video after upload - runs as async task"""
        try:
            logger.info(f"🔄 Background processing started for video: {video_name} (ID: {video_id})")

            # Wait for indexing to complete
            index_data = await self.wait_for_indexing(video_id)

            # Create GPT summary
            logger.info("  📝 Creating GPT summary...")
            summary_text = ""
            try:
                summary_id = self.create_textual_summary(video_id)
                summary_text = await self.get_textual_summary(video_id, summary_id)
                logger.info(f"  ✅ Received summary with length: {len(summary_text)} characters")
            except Exception as e:
                logger.info(f"  ⚠️ GPT summary creation failed, continuing without summary: {e}")

            # Extract transcript
            transcript_segments = self.extract_transcript_with_timestamps(index_data)

            # Merge segments if required
            if merge_segments_duration:
                logger.info(f"  🔗 Merging segments to maximum {merge_segments_duration} seconds...")
                transcript_segments = self.merge_segments_by_duration(transcript_segments, merge_segments_duration)

            # Extract metadata
            metadata = self.extract_video_metadata(index_data)

            # Create structured data with video name
            structured_data = {
                "id": str(file_id),  # Use file_id as identifier instead of video_id
                "name": video_name,  # Use provided name
                **metadata,
                "transcript_segments": transcript_segments,
                "full_transcript": " ".join([seg["text"] for seg in transcript_segments]),
                "segment_start_times": [seg["start_time"] for seg in transcript_segments],
                "segment_start_seconds": [seg["start_seconds"] for seg in transcript_segments],
                "summary_text": summary_text
            }

            # Convert to markdown
            md_content = self.parse_insights_to_md(structured_data)

            logger.info(f"  ✅ Processing completed successfully!")
            logger.info(f"  📊 Found {len(transcript_segments)} transcript segments")

            # Create target path and upload final result
            target_blob_path = f"{course_id}/{section_id}/Videos_md/{file_id}.md"
            blob_manager_write = BlobManager(container_name="processeddata")

            logger.info(f"📤 Uploading final result to processeddata container: {target_blob_path}")
            success = blob_manager_write.upload_text_to_blob(
                text_content=md_content,
                blob_name=target_blob_path
            )

            if success:
                logger.info(f"✅ Final file uploaded successfully to processeddata container: {target_blob_path}")
            else:
                logger.info(f"❌ Failed to upload final file to processeddata container")

            # Cleanup: delete video from Video Indexer
            logger.info("🧹 Cleaning up unnecessary containers...")
            self.delete_video(video_id)

        except Exception as e:
            logger.info(f"❌ Background video processing failed for {video_name}: {str(e)}")


async def main():
    # Process video from blob storage with new parameters
    course_id = "Information_systems"
    section_id = "Section1"
    file_id = 101
    video_name = "L1 - A "
    video_url = "A-יסודות מערכות מידע - שעור (06-11-2024) - T.mp4"

    logger.info(f"🧪 Processing video: {video_name}")
    logger.info(f"📍 CourseID: {course_id}, SectionID: {section_id}, FileID: {file_id}")
    logger.info(f"🔗 VideoURL: {video_url}")

    try:
        manager = VideoIndexerManager()
        result = await manager.process_video_to_md(course_id, section_id, file_id, video_name, video_url,
                                                   merge_segments_duration=20)

        if result:
            logger.info(f"\n🎉 Video processing started successfully: {result}")
            logger.info(f"📁 Final file will be saved to: {course_id}/{section_id}/Videos_md/{file_id}.md")
            logger.info(f"🚀 Processing continues in background...")
        else:
            logger.info(f"\n❌ Video processing failed: {video_name}")

    except Exception as e:
        logger.info(f"❌ Error processing video: {e}")

if __name__ == "__main__":
    asyncio.run(main())
