"""
Advanced Unified Content Search - מערכת חיפוש מתקדמת לתוכן מאוחד
מבוססת על האינדקס המאוחד שנוצר עם unified_indexer.py
תומכת בחיפוש בוידאו ומסמכים יחד או בנפרד
"""
import logging
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
import traceback
from Config.logging_config import setup_logging
logger = setup_logging()
from Config.config import (
    SEARCH_SERVICE_NAME, SEARCH_API_KEY,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_MODEL, INDEX_NAME
)

# Configure logging - only errors and warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedUnifiedContentSearch:
    """
    מערכת חיפוש מתקדמת לתוכן מאוחד - וידאו ומסמכים
    תומכת בחיפוש טקסטואלי, סמנטי ווקטורי
    מאפשרת חיפוש בכל התוכן יחד או בסינון לפי סוג
    """

    def __init__(self, index_name: str = INDEX_NAME):
        self.index_name = INDEX_NAME
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)

        # יצירת search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

        # יצירת OpenAI client לחיפוש וקטורי
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        logger.info(f"AdvancedUnifiedContentSearch initialized with index: {self.index_name}")

    def check_index_status(self) -> Dict:
        """בדיקת מצב האינדקס המאוחד והצגת מידע בסיסי"""
        logger.info("=" * 60)

        try:
            # חיפוש כללי לבדיקה
            results = self.search_client.search(
                search_text="*",
                select=["*"],
                top=5,
                include_total_count=True
            )

            total_count = results.get_count()
            docs = list(results)

            logger.info(f"📊 סה\"כ צ'אנקים באינדקס המאוחד: {total_count}")
            logger.info(f"📄 מסמכים שהוחזרו לבדיקה: {len(docs)}")

            if docs:
                logger.info(f"✅ האינדקס המאוחד פעיל ומכיל נתונים")

                # ספירה לפי סוג תוכן
                video_results = self.search_client.search("*", filter="content_type eq 'video'",
                                                          include_total_count=True, top=0)
                video_count = video_results.get_count()

                doc_results = self.search_client.search("*", filter="content_type eq 'document'",
                                                        include_total_count=True, top=0)
                doc_count = doc_results.get_count()

                logger.info(f"🎥 וידאו צ'אנקים: {video_count}")
                logger.info(f"📝 מסמך צ'אנקים: {doc_count}")

                # הצגת דוגמאות למסמכים
                logger.info(f"\n📄 דוגמאות למסמכים באינדקס:")
                for i, doc in enumerate(docs[:10], 1):
                    content_type = doc.get('content_type', 'unknown')
                    logger.info(f"\n📄 מסמך {i} ({content_type}):")
                    logger.info(f"  🆔 ID: {doc.get('id', 'N/A')}")
                    logger.info(f"  📄 Source ID: {doc.get('source_id', 'N/A')}")
                    logger.info(f"  📝 Source Name: {doc.get('source_name', 'N/A')}")
                    logger.info(f"  📑 Chunk Index: {doc.get('chunk_index', 'N/A')}")

                    if content_type == 'video':
                        logger.info(f"  ⏰ Start Time: {doc.get('start_time', 'N/A')}")
                        logger.info(f"  ⏱️ Start Seconds: {doc.get('start_seconds', 'N/A')}")
                    elif content_type == 'document':
                        logger.info(f"  📋 Section Title: {doc.get('section_title', 'N/A')}")
                        logger.info(f"  📄 Document Type: {doc.get('document_type', 'N/A')}")

                    # הצגת תוכן הטקסט
                    text = doc.get('text', '')
                    if text:
                        preview = text[:150] + "..." if len(text) > 150 else text
                        logger.info(f"  📜 תוכן: {preview}")
                    logger.info("-" * 30)

                return {
                    "status": "active",
                    "total_chunks": total_count,
                    "video_chunks": video_count,
                    "document_chunks": doc_count,
                    "sample_doc": docs[0] if docs else None
                }
            else:
                logger.info("⚠️ האינדקס קיים אבל ריק")
                return {"status": "empty", "total_chunks": 0}

        except Exception as e:
            logger.info(f"❌ שגיאה בגישה לאינדקס: {e}")
            logger.error(f"Error checking index status: {e}")
            return {"status": "error", "error": str(e)}

    def generate_query_embedding(self, query: str) -> List[float]:
        """יצירת embedding לשאלת החיפוש"""
        try:
            response = self.openai_client.embeddings.create(
                model=AZURE_OPENAI_EMBEDDING_MODEL,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    def simple_text_search(self, query: str, top_k: int = 5, content_type: str = None, source_id: str = None) -> List[
        Dict]:
        """חיפוש טקסטואלי פשוט במידה ולא ניתן לחלץ embedding"""
        logger.info("=" * 60)

        try:
            search_params = {
                "search_text": query,
                "select": [
                    "id", "content_type", "source_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics"
                ],
                "top": top_k,
                "include_total_count": True
            }

            # הוספת פילטרים
            filters = []
            if content_type:
                filters.append(f"content_type eq '{content_type}'")
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            results = self.search_client.search(**search_params)

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                logger.info("❌ לא נמצאו תוצאות")
                return []

            filter_msg = self._build_filter_message(content_type, source_id)
            logger.info(f"✅ נמצאו {len(docs)} תוצאות מתוך {total_count} צ'אנקים{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\n📄 תוצאה {i} ({content_type_doc}, ציון: {score:.3f}):")
                logger.info(f"  📄 מקור ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  ⏰ זמן: {start_time} - {end_time}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  📜 תוכן: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"❌ שגיאה בחיפוש טקסטואלי: {e}")
            logger.error(f"Error in text search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5, content_type: str = None, source_id: str = None) -> List[Dict]:
        """חיפוש היברידי - משלב טקסט ווקטור"""
        logger.info("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                logger.info("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k, content_type, source_id)

            search_params = {
                "search_text": query,
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=50,
                    fields="vector"
                )],
                "select": [
                    "id", "content_type", "source_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics"
                ],
                "top": 50,
                "include_total_count": True
            }

            # הוספת פילטרים
            filters = []
            if content_type:
                filters.append(f"content_type eq '{content_type}'")
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            results = self.search_client.search(**search_params)

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                logger.info("❌ לא נמצאו תוצאות היברידיות")
                return []

            # Slice to requested top_k for display and return
            docs = docs[:top_k]

            filter_msg = self._build_filter_message(content_type, source_id)
            logger.info(f"✅ נמצאו {len(docs)} תוצאות היברידיות מתוך {total_count} צ'אנקים{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\n📄 תוצאה {i} ({content_type_doc}, ציון משולב: {score:.3f}):")
                logger.info(f"  📄 מקור ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  ⏰ זמן: {start_time} - {end_time}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  📜 תוכן: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"❌ שגיאה בחיפוש היברידי: {e}")
            logger.error(f"Error in hybrid search: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 5, content_type: str = None, source_id: str = None) -> List[
        Dict]:
        """חיפוש סמנטי מתקדם"""
        logger.info("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                logger.info("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k, content_type, source_id)

            # הכנת פרמטרים לחיפוש
            search_params = {
                "search_text": query,
                "query_type": "semantic",
                "semantic_configuration_name": "default",
                "query_language": "he-il",
                "highlight_fields": "text",
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )],
                "select": [
                    "id", "content_type", "source_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics"
                ],
                "top": top_k
            }

            # הוספת פילטרים
            filters = []
            if content_type:
                filters.append(f"content_type eq '{content_type}'")
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            # חיפוש סמנטי מתקדם
            results = self.search_client.search(**search_params)

            docs = list(results)

            if not docs:
                logger.info("❌ לא נמצאו תוצאות סמנטיות")
                return []

            filter_msg = self._build_filter_message(content_type, source_id)
            logger.info(f"✅ נמצאו {len(docs)} תוצאות סמנטיות{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\n📄 תוצאה {i} ({content_type_doc}, ציון סמנטי: {score:.3f}):")
                logger.info(f"  📄 מקור ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  ⏰ זמן: {start_time} - {end_time}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  📜 תוכן: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"❌ שגיאה בחיפוש סמנטי מתקדם: {e}")
            logger.error(f"Error in semantic search: {e}")
            # Fallback to regular hybrid search
            return self.hybrid_search(query, top_k, content_type, source_id)

    def search_best_answers(self, query: str, k: int = 5, content_type: str = None, source_id: str = None) -> List[
        Dict]:
        """
        פונקציה פשוטה שמקבלת שאלה ומחזירה K התשובות הטובות ביותר
        משתמשת בחיפוש סמנטי כברירת מחדל, עם fallback להיברידי
        מחזירה רק את הנתונים ללא הדפסות

        Args:
            query: השאלה לחיפוש
            k: מספר התוצאות הטובות ביותר להחזיר
            content_type: אופציונלי - "video" או "document" לסינון לפי סוג תוכן
            source_id: אופציונלי - אם מוגדר, יחפש רק במקור הספציפי הזה
        """
        try:
            # שימוש בחיפוש סמנטי שהוא הכי חכם
            results = self.semantic_search(query, k, content_type, source_id)
            return results
        except Exception:
            # fallback להיברידי אם הסמנטי נכשל
            results = self.hybrid_search(query, k, content_type, source_id)
            return results

    def search_videos_only(self, query: str, k: int = 5, video_id: str = None) -> List[Dict]:
        """חיפוש רק בוידאו - תאימות לאחור עם הקובץ המקורי"""
        return self.search_best_answers(query, k, content_type="video", source_id=video_id)

    def search_documents_only(self, query: str, k: int = 5, document_id: str = None) -> List[Dict]:
        """חיפוש רק במסמכים"""
        return self.search_best_answers(query, k, content_type="document", source_id=document_id)

    def get_video_transcript(self, video_id: str) -> Dict:
        """קבלת תמלול מלא של הרצאה ספציפית לפי ID - כמו בקובץ המקורי"""
        logger.info(f"📹 קבלת תמלול מלא עבור וידאו ID: {video_id}")
        logger.info("=" * 60)

        try:
            # קבלת כל הצ'אנקים של הוידאו
            results = self.search_client.search(
                search_text="*",
                filter=f"source_id eq '{video_id}' and content_type eq 'video'",
                select=["start_time", "end_time", "text", "created_date", "chunk_index"],
                order_by=["chunk_index asc"],
                top=1000  # מספר גבוה לקבלת כל הצ'אנקים
            )

            docs = list(results)

            if not docs:
                logger.info(f"❌ לא נמצא וידאו עם ID: {video_id}")
                return {"error": f"לא נמצא וידאו עם ID: {video_id}"}

            logger.info(f"🎬 וידאו ID: {video_id}")
            logger.info(f"📄 סה\"כ צ'אנקים: {len(docs)}")
            logger.info("\n📜 תמלול מלא:")
            logger.info("=" * 60)

            # הדפסת התמלול המלא
            for i, doc in enumerate(docs, 1):
                start_time = doc.get('start_time', 'N/A')
                end_time = doc.get('end_time', 'N/A')
                text = doc.get('text', '')

                logger.info(f"\n⏰ [{start_time} - {end_time}]")
                logger.info(f"{text}")
                logger.info("-" * 40)

            # חישוב סטטיסטיקות
            total_chunks = len(docs)
            total_text_length = sum(len(doc.get('text', '')) for doc in docs)

            summary = {
                "video_id": video_id,
                "total_chunks": total_chunks,
                "total_text_characters": total_text_length,
                "average_chunk_length": round(total_text_length / total_chunks) if total_chunks > 0 else 0,
                "created_date": docs[0].get('created_date') if docs else None,
                "transcript": docs
            }

            logger.info(f"\n📊 סיכום:")
            logger.info(f"  📝 וידאו ID: {summary['video_id']}")
            logger.info(f"  📄 צ'אנקים: {summary['total_chunks']}")
            logger.info(f"  📏 אורך טקסט כולל: {summary['total_text_characters']} תווים")

            return summary

        except Exception as e:
            logger.info(f"❌ שגיאה בקבלת תמלול: {e}")
            logger.error(f"Error getting video transcript: {e}")
            return {"error": str(e)}

    def _build_filter_message(self, content_type: str = None, source_id: str = None) -> str:
        """בניית הודעת פילטר לתצוגה"""
        filter_parts = []
        if content_type:
            filter_parts.append(f"סוג: {content_type}")
        if source_id:
            filter_parts.append(f"מקור: {source_id}")

        if filter_parts:
            return f" (מסונן ל-{', '.join(filter_parts)})"
        return ""

    def get_content_summary(self, source_id: str, content_type: str = None) -> Dict:
        """קבלת סיכום של מקור ספציפי (וידאו או מסמך)"""
        try:
            # בניית פילטר
            filters = [f"source_id eq '{source_id}'"]
            if content_type:
                filters.append(f"content_type eq '{content_type}'")

            filter_str = " and ".join(filters)

            # קבלת כל הצ'אנקים של המקור
            results = self.search_client.search(
                search_text="*",
                filter=filter_str,
                select=["content_type", "text", "section_title", "created_date", "chunk_index"],
                order_by=["chunk_index asc"],
                top=1000  # מספר גבוה לקבלת כל הצ'אנקים
            )


            docs = list(results)

            if not docs:
                return {"error": f"לא נמצא תוכן עם ID: {source_id}"}

            # חישוב סטטיסטיקות
            total_chunks = len(docs)
            detected_content_type = docs[0].get('content_type', 'unknown')
            total_text_length = sum(len(doc.get('text', '')) for doc in docs)

            summary = {
                "source_id": source_id,
                "content_type": detected_content_type,
                "total_chunks": total_chunks,
                "total_text_characters": total_text_length,
                "average_chunk_length": round(total_text_length / total_chunks) if total_chunks > 0 else 0,
                "created_date": docs[0].get('created_date') if docs else None
            }

            # הוספת מידע ספציפי לסוג התוכן
            if detected_content_type == 'document':
                # ספירת סעיפים
                sections = set(doc.get('section_title', '') for doc in docs if doc.get('section_title'))
                summary["unique_sections"] = len(sections)
                summary["section_titles"] = list(sections)

            logger.info(f"📊 סיכום {detected_content_type} {source_id}:")
            logger.info(f"  📝 מקור ID: {source_id}")
            logger.info(f"  📄 צ'אנקים: {summary['total_chunks']}")
            logger.info(f"  📏 אורך טקסט כולל: {summary['total_text_characters']} תווים")
            logger.info(f"  📊 אורך צ'אנק ממוצע: {summary['average_chunk_length']} תווים")

            if detected_content_type == 'document' and 'unique_sections' in summary:
                logger.info(f"  📋 סעיפים: {summary['unique_sections']}")

            return summary

        except Exception as e:
            logger.error(f"Error creating content summary: {e}")
            return {"error": str(e)}


def run_unified_search_demo():
    """הרצת דמו מלא של מערכת החיפוש המאוחדת"""
    logger.info("🔍 מערכת חיפוש מתקדמת לתוכן מאוחד - וידאו ומסמכים")
    logger.info("=" * 80)

    try:
        # יצירת מערכת החיפוש
        search_system = AdvancedUnifiedContentSearch("unified-content-chunks")

        # בדיקת מצב האינדקס
        logger.info("\n🔧 בדיקת מצב האינדקס המאוחד:")
        status = search_system.check_index_status()

        if status.get("status") != "active":
            logger.info("❌ האינדקס לא פעיל או ריק. אנא ודא שהאינדקס נוצר ומכיל נתונים.")
            return

        # שאלות לדוגמה
        demo_queries = [
            "מה זה טרנזטיביות",
            "מתי יש שוויון בין מחלקות שקילות",
            "איך אפשר לשלול ביטוי"
        ]

        logger.info(f"\n🎯 הרצת דמו עם {len(demo_queries)} שאלות:")

        for i, query in enumerate(demo_queries, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🔢 שאלה {i} מתוך {len(demo_queries)}: '{query}'")
            logger.info(f"{'=' * 80}")

            # 1. חיפוש בכל התוכן
            logger.info(f"\n1️⃣ חיפוש בכל התוכן (וידאו + מסמכים):")
            logger.info("-" * 50)
            search_system.semantic_search(query, top_k=3)

            logger.info("\n" + "=" * 80)

            # 2. חיפוש רק בוידאו
            logger.info(f"\n2️⃣ חיפוש רק בוידאו:")
            logger.info("-" * 30)
            search_system.semantic_search(query, top_k=2, content_type="video")

            logger.info("\n" + "=" * 80)

            # 3. חיפוש רק במסמכים
            logger.info(f"\n3️⃣ חיפוש רק במסמכים:")
            logger.info("-" * 35)
            search_system.semantic_search(query, top_k=2, content_type="document")

            logger.info("\n" + "=" * 80)

            # 4. חיפוש בוידאו ספציפי - כמו בקובץ המקורי
            logger.info(f"\n4️⃣ חיפוש בוידאו ספציפי:")
            logger.info("-" * 40)
            # נניח שיש לנו וידאו עם ID זה (תצטרך להחליף לID אמיתי)
            sample_video_id = "2"
            logger.info(f"🎯 חיפוש בוידאו: {sample_video_id}")
            search_system.semantic_search(query, top_k=2, content_type="video", source_id=sample_video_id)

            # הפסקה בין שאלות
            if i < len(demo_queries):
                logger.info("\n" + "🔄 עובר לשאלה הבאה..." + "\n")

        # סיכום תוכן לדוגמה
        if status.get("sample_doc"):
            sample_source_id = status["sample_doc"].get("source_id")
            sample_content_type = status["sample_doc"].get("content_type")
            if sample_source_id:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"📊 סיכום התוכן לדוגמה")
                logger.info(f"{'=' * 80}")
                search_system.get_content_summary(sample_source_id, sample_content_type)

        logger.info(f"\n🎉 דמו הושלם בהצלחה!")
        logger.info("💡 ניתן להשתמש במערכת עם שאלות נוספות או לחפש לפי סוג תוכן ספציפי")

    except Exception as e:
        logger.info(f"❌ שגיאה בהרצת הדמו: {e}")
        logger.error(f"Error in demo: {e}")
        traceback.logger.info_exc()


def main():
    """פונקציה ראשית"""
    try:
        # הרצת דמו
        run_unified_search_demo()

        # יצירת מערכת החיפוש
        search_system = AdvancedUnifiedContentSearch("unified-content-chunks")

        # שאלה לדוגמה
        query = "מה ההגדרה של יחס שקילות"

        logger.info(f"\n🔍 חיפוש מאוחד עבור: '{query}'")
        logger.info("=" * 60)

        # חיפוש פשוט אחד בלבד
        results = search_system.search_best_answers(query, k=5)

        if not results:
            logger.info("❌ לא נמצאו תוצאות")
            return

        logger.info(f"\n📋 {len(results)} התשובות הטובות ביותר:")
        logger.info("=" * 60)

        for i, doc in enumerate(results, 1):
            score = doc.get('@search.score', 0)
            content_type = doc.get('content_type', 'unknown')
            logger.info(f"\n🏆 תשובה {i} ({content_type}, ציון: {score:.3f}):")
            logger.info(f"  📄 מקור: {doc.get('source_name', 'לא זמין')}")
            logger.info(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

            if content_type == 'video':
                start_time = doc.get('start_time', '')
                if start_time:
                    logger.info(f"  ⏰ זמן: {start_time}")
            elif content_type == 'document':
                section_title = doc.get('section_title', '')
                if section_title:
                    logger.info(f"  📋 כותרת: {section_title}")

            text = doc.get('text', '')
            if text:
                logger.info(f"  💬 תוכן: {text}")

            logger.info("-" * 40)

    except Exception as e:
        logger.info(f"❌ שגיאה כללית: {e}")
        traceback.logger.info_exc()


if __name__ == "__main__":
    main()
