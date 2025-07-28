"""
Blob Storage Manager
מנהל העלאה והורדה של קבצים ל-Azure Blob Storage
תומך בכל סוגי הקבצים: MP4, MD, PDF, JSON וכו'
"""

import os
from typing import Optional, List
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from Config.config import STORAGE_CONNECTION_STRING, CONTAINER_NAME
import traceback
from Config.logging_config import setup_logging
logger = setup_logging()

class BlobManager:
    """מנהל העלאה והורדה של קבצים ל-Azure Blob Storage"""

    def __init__(self, container_name: str = None):
        self.blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        # אם לא הועבר container_name, השתמש בדיפולטיבי מהקונפיג
        self.container_name = container_name if container_name is not None else CONTAINER_NAME

        # מיפוי סוגי קבצים ל-content types
        # מיפוי סוגי קבצים ל-content types
        self.content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.json': 'application/json',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.zip': 'application/zip',
            '.rar': 'application/x-rar-compressed',
            '.7z': 'application/x-7z-compressed'
        }

    def _get_content_type(self, file_path: str) -> str:
        """קביעת content type על פי סיומת הקובץ"""
        _, ext = os.path.splitext(file_path.lower())
        return self.content_types.get(ext, 'application/octet-stream')

    def download_file(self, blob_name: str, local_file_path: str) -> bool:
        """
        הורדת קובץ מ-blob storage

        Args:
            blob_name: שם הקובץ ב-blob storage (כולל תיקייה אם יש)
            local_file_path: נתיב השמירה המקומי

        Returns:
            True אם ההורדה הצליחה, False אחרת
        """
        try:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)

            logger.info(f"📥 מוריד קובץ: {blob_name} -> {local_file_path}")

            with open(local_file_path, 'wb') as file_data:
                download_stream = blob_client.download_blob()
                file_data.write(download_stream.readall())

            logger.info(f"✅ הקובץ הורד בהצלחה: {local_file_path}")
            return True

        except Exception as e:
            logger.info(f"❌ שגיאה בהורדת הקובץ {blob_name}: {e}")
            return False

    def list_files(self, folder: Optional[str] = None) -> List[str]:
        """
        רשימת קבצים ב-blob storage

        Args:
            folder: תיקייה ספציפית (אופציונלי - אם לא מוגדר יציג את כל הקבצים)

        Returns:
            רשימת שמות הקבצים
        """
        try:
            container_client = self.blob_service.get_container_client(self.container_name)

            if folder:
                blobs = container_client.list_blobs(name_starts_with=f"{folder}/")
            else:
                blobs = container_client.list_blobs()

            blob_list = []
            for blob in blobs:
                blob_list.append(blob.name)

            return blob_list

        except Exception as e:
            logger.info(f"❌ שגיאה ברשימת קבצים: {e}")
            return []

    # def download_folder_files(self, blob_folder_path: str, local_temp_dir: str) -> List[str]:
    #     """
    #     הורדת כל הקבצים מתיקייה ב-blob storage לתיקייה מקומית זמנית
    #
    #     Args:
    #         blob_folder_path: נתיב התיקייה ב-blob storage (למשל: "Raw-data/Docs")
    #         local_temp_dir: תיקייה מקומית זמנית לשמירת הקבצים
    #
    #     Returns:
    #         רשימת נתיבי הקבצים המקומיים שהורדו
    #     """
    #     try:
    #         container_client = self.blob_service.get_container_client(self.container_name)
    #
    #         # יצירת התיקייה המקומית אם לא קיימת
    #         os.makedirs(local_temp_dir, exist_ok=True)
    #
    #         downloaded_files = []
    #
    #         logger.info(f"🌐 מוריד קבצים מ-blob: {blob_folder_path}")
    #
    #         # רשימת כל הקבצים בתיקייה
    #         blob_list = container_client.list_blobs(name_starts_with=blob_folder_path)
    #
    #         for blob in blob_list:
    #             # דילוג על תיקיות (שמות שמסתיימים ב-/)
    #             if blob.name.endswith('/'):
    #                 continue
    #
    #             # קבלת שם הקובץ בלבד (ללא הנתיב המלא)
    #             filename = os.path.basename(blob.name)
    #             local_file_path = os.path.join(local_temp_dir, filename)
    #
    #             logger.info(f"📥 מוריד: {blob.name} → {local_file_path}")
    #
    #             # הורדת הקובץ
    #             blob_client = container_client.get_blob_client(blob.name)
    #             with open(local_file_path, "wb") as download_file:
    #                 download_file.write(blob_client.download_blob().readall())
    #
    #             downloaded_files.append(local_file_path)
    #
    #         logger.info(f"✅ הורדו {len(downloaded_files)} קבצים מ-blob storage")
    #         return downloaded_files
    #
    #     except Exception as e:
    #         logger.info(f"❌ שגיאה בהורדת קבצים מ-blob storage: {e}")
    #         return []

    def generate_sas_url(self, blob_name: str, hours: int = 4) -> str:
        """
        יצירת SAS URL לקריאה של קובץ ב-blob storage

        Args:
            blob_name: שם הקובץ ב-blob storage (כולל תיקייה אם יש)
            hours: מספר שעות שה-SAS יהיה תקף (ברירת מחדל: 4 שעות)

        Returns:
            SAS URL לקובץ
        """
        try:
            # יצירת SAS token
            sas_token = generate_blob_sas(
                account_name=self.blob_service.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.blob_service.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=hours)
            )

            # יצירת URL מלא
            blob_url = f"{self.blob_service.primary_endpoint}{self.container_name}/{blob_name}?{sas_token}"

            logger.info(f"🔗 נוצר SAS URL לקובץ: {blob_name} (תקף ל-{hours} שעות)")
            return blob_url

        except Exception as e:
            logger.info(f"❌ שגיאה ביצירת SAS URL עבור {blob_name}: {e}")
            return ""

    def download_to_memory(self, blob_name: str) -> Optional[bytes]:
        """
        הורדת קובץ מ-blob storage ישירות לזיכרון

        Args:
            blob_name: שם הקובץ ב-blob storage (כולל תיקייה אם יש)

        Returns:
            bytes של הקובץ או None אם נכשל
        """
        try:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)

            logger.info(f"📥 מוריד קובץ לזיכרון: {blob_name}")

            download_stream = blob_client.download_blob()
            file_bytes = download_stream.readall()

            logger.info(f"✅ הקובץ הורד בהצלחה לזיכרון: {blob_name} ({len(file_bytes)} bytes)")
            return file_bytes

        except Exception as e:
            logger.info(f"❌ שגיאה בהורדת הקובץ לזיכרון {blob_name}: {e}")
            return None

    def upload_text_to_blob(self, text_content: str, blob_name: str, container: str = None) -> bool:
        """
        העלאת תוכן טקסט ישירות ל-blob storage ללא יצירת קובץ זמני

        Args:
            text_content: התוכן הטקסטואלי להעלאה
            blob_name: שם הקובץ ב-blob (כולל נתיב וירטואלי)
            container: שם הקונטיינר (אם לא מוגדר, ישתמש בברירת המחדל)

        Returns:
            True אם ההעלאה הצליחה, False אחרת
        """
        try:
            # שימוש בקונטיינר שהועבר או בברירת המחדל
            target_container = container if container else self.container_name

            # קביעת content type על פי סיומת הקובץ
            content_type = self._get_content_type(blob_name)

            container_client = self.blob_service.get_container_client(target_container)

            logger.info(f"📤 מעלה טקסט ל-blob: {target_container}/{blob_name}")

            # העלאה ישירה של הטקסט
            container_client.upload_blob(
                name=blob_name,
                data=text_content.encode('utf-8'),
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )

            logger.info(f"✅ הטקסט הועלה בהצלחה: {target_container}/{blob_name}")
            return True

        except Exception as e:
            logger.info(f"❌ שגיאה בהעלאת הטקסט ל-blob: {e}")
            return False

if __name__ == "__main__":
    # Test the blob manager with your specific container structure
    logger.info("🧪 Testing Blob Manager - Course Container")
    logger.info("=" * 50)

    try:
        blob_manager = BlobManager()

        # Check Section1 folder specifically
        logger.info("\n📁 קבצים בתיקיית 'Section1':")
        section1_blobs = blob_manager.list_files("Section1")
        logger.info(f"נמצאו {len(section1_blobs)} קבצים ב-Section1:")
        for blob in section1_blobs:
            logger.info(f"  - {blob}")

        logger.info("\n📁 כל הקבצים בקונטיינר:")
        all_blobs = blob_manager.list_files()
        logger.info(f"סה\"כ נמצאו {len(all_blobs)} קבצים:")
        for blob in all_blobs[:10]:  # Show first 10
            logger.info(f"  - {blob}")
        if len(all_blobs) > 10:
            logger.info(f"  ... ועוד {len(all_blobs) - 10} קבצים")

        # Test uploading a file to Section1
        logger.info("\n📤 בדיקת העלאת קובץ לתיקיית Section1:")

        # Create a test file
        test_file_path = "test_upload.txt"
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("זהו קובץ בדיקה שהועלה ל-Section1\nתאריך: 2025-01-07")

        # Upload the test file
        success = blob_manager.upload_file(test_file_path, "test_file.txt", "Section1")

        if success:
            logger.info("✅ הקובץ הועלה בהצלחה!")

            # List files again to see the new file
            logger.info("\n📁 קבצים ב-Section1 אחרי ההעלאה:")
            updated_blobs = blob_manager.list_files("Section1")
            for blob in updated_blobs:
                logger.info(f"  - {blob}")

        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

        logger.info("\n✅ Blob manager test completed!")

    except Exception as e:
        logger.info(f"❌ Failed to test blob manager: {e}")
        traceback.logger.info_exc()
