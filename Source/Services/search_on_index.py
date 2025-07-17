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
        print("=" * 60)

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

            print(f"📊 סה\"כ צ'אנקים באינדקס המאוחד: {total_count}")
            print(f"📄 מסמכים שהוחזרו לבדיקה: {len(docs)}")

            if docs:
                print(f"✅ האינדקס המאוחד פעיל ומכיל נתונים")

                # ספירה לפי סוג תוכן
                video_results = self.search_client.search("*", filter="content_type eq 'video'",
                                                          include_total_count=True, top=0)
                video_count = video_results.get_count()

                doc_results = self.search_client.search("*", filter="content_type eq 'document'",
                                                        include_total_count=True, top=0)
                doc_count = doc_results.get_count()

                print(f"🎥 וידאו צ'אנקים: {video_count}")
                print(f"📝 מסמך צ'אנקים: {doc_count}")

                # הצגת דוגמאות למסמכים
                print(f"\n📄 דוגמאות למסמכים באינדקס:")
                for i, doc in enumerate(docs[:10], 1):
                    content_type = doc.get('content_type', 'unknown')
                    print(f"\n📄 מסמך {i} ({content_type}):")
                    print(f"  🆔 ID: {doc.get('id', 'N/A')}")
                    print(f"  📄 Source ID: {doc.get('source_id', 'N/A')}")
                    print(f"  📝 Source Name: {doc.get('source_name', 'N/A')}")
                    print(f"  📑 Chunk Index: {doc.get('chunk_index', 'N/A')}")

                    if content_type == 'video':
                        print(f"  ⏰ Start Time: {doc.get('start_time', 'N/A')}")
                        print(f"  ⏱️ Start Seconds: {doc.get('start_seconds', 'N/A')}")
                    elif content_type == 'document':
                        print(f"  📋 Section Title: {doc.get('section_title', 'N/A')}")
                        print(f"  📄 Document Type: {doc.get('document_type', 'N/A')}")

                    # הצגת תוכן הטקסט
                    text = doc.get('text', '')
                    if text:
                        preview = text[:150] + "..." if len(text) > 150 else text
                        print(f"  📜 תוכן: {preview}")
                    print("-" * 30)

                return {
                    "status": "active",
                    "total_chunks": total_count,
                    "video_chunks": video_count,
                    "document_chunks": doc_count,
                    "sample_doc": docs[0] if docs else None
                }
            else:
                print("⚠️ האינדקס קיים אבל ריק")
                return {"status": "empty", "total_chunks": 0}

        except Exception as e:
            print(f"❌ שגיאה בגישה לאינדקס: {e}")
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
        print("=" * 60)

        try:
            search_params = {
                "search_text": query,
                "select": [
                    "id", "content_type", "source_id", "source_name", "chunk_index",
                    "text", "start_time", "start_seconds", "section_title", "document_type", "created_date"
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
                print("❌ לא נמצאו תוצאות")
                return []

            filter_msg = self._build_filter_message(content_type, source_id)
            print(f"✅ נמצאו {len(docs)} תוצאות מתוך {total_count} צ'אנקים{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                print(f"\n📄 תוצאה {i} ({content_type_doc}, ציון: {score:.3f}):")
                print(f"  📄 מקור: {doc.get('source_name', 'לא זמין')} (ID: {doc.get('source_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    start_seconds = doc.get('start_seconds', 0)
                    if start_time:
                        print(f"  ⏰ זמן: {start_time} ({start_seconds:.1f}s)")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש טקסטואלי: {e}")
            logger.error(f"Error in text search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5, content_type: str = None, source_id: str = None) -> List[Dict]:
        """חיפוש היברידי - משלב טקסט ווקטור"""
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k, content_type, source_id)

            search_params = {
                "search_text": query,
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=50,
                    fields="vector"
                )],
                "select": [
                    "id", "content_type", "source_id", "source_name", "chunk_index",
                    "text", "start_time", "start_seconds", "section_title", "document_type", "created_date"
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
                print("❌ לא נמצאו תוצאות היברידיות")
                return []

            # Slice to requested top_k for display and return
            docs = docs[:top_k]

            filter_msg = self._build_filter_message(content_type, source_id)
            print(f"✅ נמצאו {len(docs)} תוצאות היברידיות מתוך {total_count} צ'אנקים{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                print(f"\n📄 תוצאה {i} ({content_type_doc}, ציון משולב: {score:.3f}):")
                print(f"  📄 מקור: {doc.get('source_name', 'לא זמין')} (ID: {doc.get('source_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    start_seconds = doc.get('start_seconds', 0)
                    if start_time:
                        print(f"  ⏰ זמן: {start_time} ({start_seconds:.1f}s)")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש היברידי: {e}")
            logger.error(f"Error in hybrid search: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 5, content_type: str = None, source_id: str = None) -> List[
        Dict]:
        """חיפוש סמנטי מתקדם"""
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
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
                    "id", "content_type", "source_id", "source_name", "chunk_index",
                    "text", "start_time", "start_seconds", "section_title", "document_type", "created_date"
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
                print("❌ לא נמצאו תוצאות סמנטיות")
                return []

            filter_msg = self._build_filter_message(content_type, source_id)
            print(f"✅ נמצאו {len(docs)} תוצאות סמנטיות{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                print(f"\n📄 תוצאה {i} ({content_type_doc}, ציון סמנטי: {score:.3f}):")
                print(f"  📄 מקור: {doc.get('source_name', 'לא זמין')} (ID: {doc.get('source_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    start_seconds = doc.get('start_seconds', 0)
                    if start_time:
                        print(f"  ⏰ זמן: {start_time} ({start_seconds:.1f}s)")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש סמנטי מתקדם: {e}")
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
        print(f"📹 קבלת תמלול מלא עבור וידאו ID: {video_id}")
        print("=" * 60)

        try:
            # קבלת כל הצ'אנקים של הוידאו
            results = self.search_client.search(
                search_text="*",
                filter=f"source_id eq '{video_id}' and content_type eq 'video'",
                select=["source_name", "start_seconds", "start_time", "text", "created_date", "chunk_index"],
                order_by=["start_seconds asc"],
                top=1000  # מספר גבוה לקבלת כל הצ'אנקים
            )

            docs = list(results)

            if not docs:
                print(f"❌ לא נמצא וידאו עם ID: {video_id}")
                return {"error": f"לא נמצא וידאו עם ID: {video_id}"}

            video_name = docs[0].get('source_name', 'לא זמין')
            print(f"🎬 וידאו: {video_name}")
            print(f"📄 סה\"כ צ'אנקים: {len(docs)}")
            print("\n📜 תמלול מלא:")
            print("=" * 60)

            # הדפסת התמלול המלא
            for i, doc in enumerate(docs, 1):
                start_time = doc.get('start_time', 'N/A')
                start_seconds = doc.get('start_seconds', 0)
                text = doc.get('text', '')

                print(f"\n⏰ [{start_time}] ({start_seconds:.1f}s)")
                print(f"{text}")
                print("-" * 40)

            # חישוב סטטיסטיקות
            total_chunks = len(docs)
            total_text_length = sum(len(doc.get('text', '')) for doc in docs)
            max_time = max(doc.get('start_seconds', 0) for doc in docs)
            duration_minutes = max_time / 60

            summary = {
                "video_id": video_id,
                "video_name": video_name,
                "total_chunks": total_chunks,
                "estimated_duration_minutes": round(duration_minutes, 2),
                "total_text_characters": total_text_length,
                "average_chunk_length": round(total_text_length / total_chunks) if total_chunks > 0 else 0,
                "created_date": docs[0].get('created_date') if docs else None,
                "transcript": docs
            }

            print(f"\n📊 סיכום:")
            print(f"  📝 שם: {summary['video_name']}")
            print(f"  📄 צ'אנקים: {summary['total_chunks']}")
            print(f"  ⏱️ משך משוער: {summary['estimated_duration_minutes']} דקות")
            print(f"  📏 אורך טקסט כולל: {summary['total_text_characters']} תווים")

            return summary

        except Exception as e:
            print(f"❌ שגיאה בקבלת תמלול: {e}")
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
                select=["content_type", "source_name", "start_seconds", "text", "section_title", "created_date"],
                order_by=["chunk_index asc"],
                top=1000  # מספר גבוה לקבלת כל הצ'אנקים
            )

            docs = list(results)

            if not docs:
                return {"error": f"לא נמצא תוכן עם ID: {source_id}"}

            # חישוב סטטיסטיקות
            total_chunks = len(docs)
            source_name = docs[0].get('source_name', 'לא זמין')
            detected_content_type = docs[0].get('content_type', 'unknown')
            total_text_length = sum(len(doc.get('text', '')) for doc in docs)

            summary = {
                "source_id": source_id,
                "source_name": source_name,
                "content_type": detected_content_type,
                "total_chunks": total_chunks,
                "total_text_characters": total_text_length,
                "average_chunk_length": round(total_text_length / total_chunks) if total_chunks > 0 else 0,
                "created_date": docs[0].get('created_date') if docs else None
            }

            # הוספת מידע ספציפי לסוג התוכן
            if detected_content_type == 'video':
                # זמן כולל (משוער)
                if docs:
                    max_time = max(doc.get('start_seconds', 0) for doc in docs if doc.get('start_seconds'))
                    duration_minutes = max_time / 60 if max_time else 0
                    summary["estimated_duration_minutes"] = round(duration_minutes, 2)
            elif detected_content_type == 'document':
                # ספירת סעיפים
                sections = set(doc.get('section_title', '') for doc in docs if doc.get('section_title'))
                summary["unique_sections"] = len(sections)
                summary["section_titles"] = list(sections)

            print(f"📊 סיכום {detected_content_type} {source_id}:")
            print(f"  📝 שם: {summary['source_name']}")
            print(f"  📄 צ'אנקים: {summary['total_chunks']}")
            print(f"  📏 אורך טקסט כולל: {summary['total_text_characters']} תווים")
            print(f"  📊 אורך צ'אנק ממוצע: {summary['average_chunk_length']} תווים")

            if detected_content_type == 'video' and 'estimated_duration_minutes' in summary:
                print(f"  ⏱️ משך משוער: {summary['estimated_duration_minutes']} דקות")
            elif detected_content_type == 'document' and 'unique_sections' in summary:
                print(f"  📋 סעיפים: {summary['unique_sections']}")

            return summary

        except Exception as e:
            logger.error(f"Error creating content summary: {e}")
            return {"error": str(e)}


def run_unified_search_demo():
    """הרצת דמו מלא של מערכת החיפוש המאוחדת"""
    print("🔍 מערכת חיפוש מתקדמת לתוכן מאוחד - וידאו ומסמכים")
    print("=" * 80)

    try:
        # יצירת מערכת החיפוש
        search_system = AdvancedUnifiedContentSearch("unified-content-chunks")

        # בדיקת מצב האינדקס
        print("\n🔧 בדיקת מצב האינדקס המאוחד:")
        status = search_system.check_index_status()

        if status.get("status") != "active":
            print("❌ האינדקס לא פעיל או ריק. אנא ודא שהאינדקס נוצר ומכיל נתונים.")
            return

        # שאלות לדוגמה
        demo_queries = [
            "מה זה טרנזטיביות",
            "מתי יש שוויון בין מחלקות שקילות",
            "איך אפשר לשלול ביטוי"
        ]

        print(f"\n🎯 הרצת דמו עם {len(demo_queries)} שאלות:")

        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"🔢 שאלה {i} מתוך {len(demo_queries)}: '{query}'")
            print(f"{'=' * 80}")

            # 1. חיפוש בכל התוכן
            print(f"\n1️⃣ חיפוש בכל התוכן (וידאו + מסמכים):")
            print("-" * 50)
            search_system.semantic_search(query, top_k=3)

            print("\n" + "=" * 80)

            # 2. חיפוש רק בוידאו
            print(f"\n2️⃣ חיפוש רק בוידאו:")
            print("-" * 30)
            search_system.semantic_search(query, top_k=2, content_type="video")

            print("\n" + "=" * 80)

            # 3. חיפוש רק במסמכים
            print(f"\n3️⃣ חיפוש רק במסמכים:")
            print("-" * 35)
            search_system.semantic_search(query, top_k=2, content_type="document")

            print("\n" + "=" * 80)

            # 4. חיפוש בוידאו ספציפי - כמו בקובץ המקורי
            print(f"\n4️⃣ חיפוש בוידאו ספציפי:")
            print("-" * 40)
            # נניח שיש לנו וידאו עם ID זה (תצטרך להחליף לID אמיתי)
            sample_video_id = "2"
            print(f"🎯 חיפוש בוידאו: {sample_video_id}")
            search_system.semantic_search(query, top_k=2, content_type="video", source_id=sample_video_id)

            # הפסקה בין שאלות
            if i < len(demo_queries):
                print("\n" + "🔄 עובר לשאלה הבאה..." + "\n")

        # סיכום תוכן לדוגמה
        if status.get("sample_doc"):
            sample_source_id = status["sample_doc"].get("source_id")
            sample_content_type = status["sample_doc"].get("content_type")
            if sample_source_id:
                print(f"\n{'=' * 80}")
                print(f"📊 סיכום התוכן לדוגמה")
                print(f"{'=' * 80}")
                search_system.get_content_summary(sample_source_id, sample_content_type)

        print(f"\n🎉 דמו הושלם בהצלחה!")
        print("💡 ניתן להשתמש במערכת עם שאלות נוספות או לחפש לפי סוג תוכן ספציפי")

    except Exception as e:
        print(f"❌ שגיאה בהרצת הדמו: {e}")
        logger.error(f"Error in demo: {e}")
        traceback.print_exc()


def main():
    """פונקציה ראשית"""
    try:
        # הרצת דמו
        run_unified_search_demo()

        # יצירת מערכת החיפוש
        search_system = AdvancedUnifiedContentSearch("unified-content-chunks")

        # שאלה לדוגמה
        query = "מה ההגדרה של יחס שקילות"

        print(f"\n🔍 חיפוש מאוחד עבור: '{query}'")
        print("=" * 60)

        # חיפוש פשוט אחד בלבד
        results = search_system.search_best_answers(query, k=5)

        if not results:
            print("❌ לא נמצאו תוצאות")
            return

        print(f"\n📋 {len(results)} התשובות הטובות ביותר:")
        print("=" * 60)

        for i, doc in enumerate(results, 1):
            score = doc.get('@search.score', 0)
            content_type = doc.get('content_type', 'unknown')
            print(f"\n🏆 תשובה {i} ({content_type}, ציון: {score:.3f}):")
            print(f"  📄 מקור: {doc.get('source_name', 'לא זמין')}")
            print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")

            if content_type == 'video':
                start_time = doc.get('start_time', '')
                if start_time:
                    print(f"  ⏰ זמן: {start_time}")
            elif content_type == 'document':
                section_title = doc.get('section_title', '')
                if section_title:
                    print(f"  📋 כותרת: {section_title}")

            text = doc.get('text', '')
            if text:
                print(f"  💬 תוכן: {text}")

            print("-" * 40)

    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
